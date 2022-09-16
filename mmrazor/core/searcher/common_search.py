# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import time
import torch

from collections import OrderedDict
import numpy as np
import mmcv.fileio
from mmcv.runner import get_dist_info

from ..builder import SEARCHERS
from ..utils import broadcast_object_list

from ...models.pruners import feature_pool,\
    METRICS,WEIGHT_METRICS,FILTER_METRICS
from cka import cka_t,gram_t
import geatpy as ea

@SEARCHERS.register_module()
class CKAEvolutionSearcher():
    """Implement of evolution search.

    Args:
        algorithm (:obj:`torch.nn.Module`): Algorithm to be used.
        dataloader (nn.Dataloader): Pytorch data loader.
        test_fn (function): Test api to used for evaluation.
        work_dir (str): Working direction is to save search result and log.
        logger (logging.Logger): To log info in search stage.
        candidate_pool_size (int): The length of candidate pool.
        candidate_top_k (int): Specify top k candidates based on scores.
        constraints (dict): Constraints to be used for screening candidates.
        metrics (str): Metrics to be used for evaluating candidates.
        metric_options (str): Options to be used for metrics.
        score_key (str): To be used for specifying one metric from evaluation
            results.
        max_epoch (int): Specify max epoch to end evolution search.
        num_mutation (int): The number of candidates got by mutation.
        num_crossover (int): The number of candidates got by crossover.
        mutate_prob (float): The probability of mutation.
        resume_from (str): Specify the path of saved .pkl file for resuming
            searching
    """

    def __init__(self,
                 algorithm,
                 dataloader,
                 test_fn,
                 work_dir,
                 logger,
                 reduction_ratio,
                 candidate_pool_size=50,
                 candidate_top_k=10,
                 constraints=dict(flops=330 * 1e6),
                 metric:str=None,
                 metric_options:list=None,
                 score_key='accuracy_top-1',
                 max_epoch=20,
                 num_mutation=25,
                 num_crossover=25,
                 mutate_prob=0.1,
                 resume_from=None,
                 rand_seed=0,
                 **search_kwargs):

        if not hasattr(algorithm, 'module'):
            raise NotImplementedError('Do not support searching with cpu.')
        self.algorithm = algorithm.module
        self.algorithm_for_test = algorithm
        self.dataloader = dataloader
        self.constraints = constraints
        self.metric = metric # 单一metric剪枝使用，定义要用的剪枝标准 “”
        self.metric_options = metric_options # 混合剪枝时使用，定义要用的剪枝标准们 ["","",""]
        self.score_key = score_key
        self.candidate_pool = list()
        self.candidate_pool_size = candidate_pool_size
        self.max_epoch = max_epoch
        self.test_fn = test_fn
        self.candidate_top_k = candidate_top_k
        self.num_mutation = num_mutation # 
        self.num_crossover = num_crossover #
        self.mutate_prob = mutate_prob
        self.top_k_candidates_with_score = dict()
        self.candidate_pool_with_score = dict()
        self.work_dir = work_dir
        self.resume_from = resume_from
        self.logger = logger
        # self
        self.rand_seed=rand_seed
        self.reduction_ratio=reduction_ratio
        self.features_dict={}
        self.weights_dict={}
        self.bias_dict={}

        if self.metric :
            assert METRICS.__contains__(self.metric), f"Input Metric {self.metric} not in metrics list:{METRICS.keys()}"
        if self.metric_options:
            for m in self.metric_options:
                assert METRICS.__contains__(self.m), f"Input Metric {m} not in metrics list:{METRICS.keys()}. \
                     Total input metrics are {self.metric_options}"

    def generate_hook(self,name,fdict:dict,fpool:dict|None=None):
        def output_hook(module,input,output) -> None:
            fdict[name]=output.detach()
            if fpool:
                fpool[name]=feature_pool(output)
        return output_hook

    def extract_features(self, algorithm_for_test, dataloader, get_pool:bool=False):
        """
            Use hooks to get features from model's forward.
            Return: space2name: {k:[n1,...,nn]}
        """
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner

        space2name={}
        for name,module in supernet.model.named_modules():
            spi=pruner.get_space_id(name)
            if not isinstance(spi,str): continue
            g=space2name.setdefault(spi,[])
            g.append(name)
            space2name[spi]=g
        
        hookers=[]
        features_dict={}
        features_pool={} if get_pool else None
        names=[n for v in space2name.values() for n in v]
        for name, module in supernet.model.named_modules():
            if name in names:
                oo=self.generate_hook(name,features_dict,features_pool)
                hooker=module.register_forward_hook(oo)
                hookers.append(hooker)
                self.weights_dict[name]=module.weight
                if hasattr(module,'bias'):
                    self.bias_dict[name]=module.bias
        
        # forward a batch
        for i, data_batch in enumerate(dataloader):
            if i>0:break
            algorithm_for_test(**data_batch,return_loss=False)
        for hooker in hookers:
            hooker.remove()

        return space2name,features_dict,features_pool

    def test_search_geatpy_topk(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_=self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        # MOD-end

        # MOD-start 
        def evalVars(Vars):
            f,cv=[],[]
            for Vs in Vars:
                # 计算 sim cka
                iszero,sum_sim=False,0
                assert len(Vs)==len(space_list)
                for space,ratio in zip(space_list,Vs):
                    if iszero: break
                    for n in space2names[space]:
                        supf=self.features_dict[n] # N,C,H,W
                        if ratio < 1/supf.size(1): # ratio小于1/通道数
                            iszero=True
                            break
                        supf=supf.view(supf.size(0),-1)
                        subf=self.features_dict[n][:,:k] # 直接取前k个通道
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f0=0 if iszero else sum_sim.data.cpu()
                f.append(f0)
                # MOD-start 计算FLOPS
                if not iszero:
                    space2ratio={s:r for s,r in zip(space_list,Vs) }
                    sub_dict=pruner.sample_subnet_ratios(space2ratio)
                    pruner.set_subnet(sub_dict)
                    flops=self.algorithm.get_subnet_flops()
                    rflops=self.algorithm.get_raw_flops()
                    reduction_rate=(rflops-flops)/rflops
                    cv.append(self.reduction_ratio- reduction_rate)
                else:
                    cv.append(1)
                # MOD-end
            return np.array([f]).T, np.array([cv]).T
        # MOD-end
        # MOD-start geatpy问题定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= len(space2names),
            varTypes=[0]*len(space2names),
            lb=[0.01]*len(space2names),
            ub=[1]*len(space2names),
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='RI',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        # MOD-end
        self.logger.info(f"RESULTS are: {res}")

    def test_search_geatpy_discrete(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_=self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        # MOD-end
        # MOD-start evalVars
        def evalVars(Vars):
            f,cv=[],[]
            for Vs in Vars:
                # 计算 sim cka
                iszero,sum_sim=False,0
                assert len(Vs)==len(space_list)
                for space,cout_num in zip(space_list,Vs):
                    for n in space2names[space]:
                        supf=self.features_dict[n]
                        subf=self.features_dict[n][:,:cout_num]
                        supf=supf.view(supf.size(0),-1)
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f.append(sum_sim.data.cpu())
                # 计算FLOPS
                space2ratio={}
                for s,cn in zip(space_list,Vs):
                    n=space2names[s][0]
                    space2ratio[s]=cn/self.features_dict[n].size(1)
                sub_dict=pruner.sample_subnet_ratios(space2ratio)
                pruner.set_subnet(sub_dict)
                flops=self.algorithm.get_subnet_flops()
                rflops=self.algorithm.get_raw_flops()
                reduction_rate=(rflops-flops)/rflops
                cv.append(self.reduction_ratio- reduction_rate)

            return np.array([f]).T, np.array([cv]).T

        upperbound=[]
        for sp in space_list:
            c_out=[]
            for n in space2names[sp]:
                supf=self.features_dict[n]
                c_out.append(int(supf.size(1))) # N,C,H,W
            c_out=list(set(c_out))
            assert len(c_out)<= 1,f'group{sp} channels not same:{c_out}'
            upperbound.append(c_out[0])
        # MOD-start geatpy问题离散定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= len(space2names),
            varTypes=[1]*len(space2names),
            lb=[1]*len(space2names),
            ub=upperbound,
            ubin=[0]*len(space2names),
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='BG',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        self.logger.info(f"RESULTS are: {res}")
        # MOD-end
        
    def test_search_geatpy_discrete_metric(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        assert self.metric, "This function needs denotion of searcher.metric"
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_= self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        # MOD-end
        assert self.metric in METRICS.keys(), \
            f"Metric {self.metric} not in M types:{METRICS.keys()}"

        vdict= self.features_dict if self.metric in METRICS.keys() else self.weights_dict
        filter_rank={k:METRICS[self.metric](v) for k,v in vdict.items()}

        # MOD-start evalVars
        def evalVars(Vars):
            f,cv=[],[]
            for Vs in Vars:
                # 计算 sim cka
                iszero,sum_sim=False,0
                assert len(Vs)==len(space_list)
                for space,cout_num in zip(space_list,Vs):
                    for n in space2names[space]:
                        supf=self.features_dict[n]
                        device=self.features_dict[n].device
                        ind=torch.topk(filter_rank[n],cout_num)[1].to(device)
                        subf=self.features_dict[n].index_select(1,ind) # 第1维，ind下标的通道(N,C,H,W)
                        supf=supf.view(supf.size(0),-1)
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f.append(sum_sim.data.cpu())
                # 计算FLOPS
                space2ratio={}
                for s,cn in zip(space_list,Vs):
                    n=space2names[s][0]
                    space2ratio[s]=cn/self.features_dict[n].size(1)
                sub_dict=pruner.sample_subnet_ratios(space2ratio)
                pruner.set_subnet(sub_dict)
                flops=self.algorithm.get_subnet_flops()
                rflops=self.algorithm.get_raw_flops()
                reduction_rate=(rflops-flops)/rflops
                cv.append(self.reduction_ratio- reduction_rate)

            return np.array([f]).T, np.array([cv]).T
        # 离散化情况下，每个space的上界都不同
        upperbound=[]
        for sp in space_list:
            c_out=[]
            for n in space2names[sp]:
                supf=self.features_dict[n]
                c_out.append(int(supf.size(1))) # N,C,H,W
            c_out=list(set(c_out))
            assert len(c_out)<= 1,f'group{sp} channels not same:{c_out}'
            upperbound.append(c_out[0])
        # MOD-start geatpy问题离散定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= len(space2names),
            varTypes=[1]*len(space2names),
            lb=[1]*len(space2names),
            ub=upperbound,
            ubin=[0]*len(space2names),
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='BG',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        self.logger.info(f"RESULTS are: {res}")
        # MOD-end
        
    def test_search_geatpy_discrete_hybrid(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        assert self.metric_options, f"Hybrid search must get a list of metrics!"
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_= self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        name_list=list(self.features_dict.keys())
        # MOD-end
        self.logger.info("NAME2RANK start")
        filter_ranks={k:{} for k in self.metric_options}
        for mn in self.metric_options: # 遍历指定的metrics类型
            if FILTER_METRICS.__contains__(mn): # 如果是基于filter(N,C,H,W)的
                vd=self.features_dict
            elif WEIGHT_METRICS.__contains__(mn): # 如果是基于weight(C_out,C_in,W,W)的
                vd=self.weights_dict
            else: assert False,f"Metric:{mn} not contained in F{FILTER_METRICS.keys()},or W{WEIGHT_METRICS.keys()}"
            for k,v in vd.items():
                filter_ranks[mn][k]=METRICS[mn](v) # 计算 mn剪枝标准下，名为k的module的排序
        self.logger.info("NAME2RANK finished")
        # MOD-start evalVars
        def evalVars(Vars):
            f,cv=[],[]
            for Vm in Vars:
                # 计算 sim cka
                iszero,sum_sim=False,0
                Vs=Vm[:len(space_list)]
                Ms=Vm[len(space_list):]
                assert len(Vs)==len(space_list)
                for space,cout_num in zip(space_list,Vs):
                    for n in space2names[space]:
                        supf=self.features_dict[n]
                        device=self.features_dict[n].device
                        i=name_list.index(n) # 查找名为n的下标i
                        mn= self.metric_options[Ms[i]] #  选取Ms的第i个数字，该数字是metric_options的下标，取得剪枝标准名
                        ind=torch.topk(filter_ranks[mn][n],cout_num)[1].to(device)
                        subf=self.features_dict[n].index_select(1,ind) # 第1维，ind下标的通道(N,C,H,W)
                        supf=supf.view(supf.size(0),-1)
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f.append(sum_sim.data.cpu())
                # 计算FLOPS
                space2ratio={}
                for s,cn in zip(space_list,Vs):
                    n=space2names[s][0]
                    space2ratio[s]=cn/self.features_dict[n].size(1)
                sub_dict=pruner.sample_subnet_ratios(space2ratio)
                pruner.set_subnet(sub_dict)
                flops=self.algorithm.get_subnet_flops()
                rflops=self.algorithm.get_raw_flops()
                reduction_rate=(rflops-flops)/rflops
                cv.append(self.reduction_ratio- reduction_rate)

            return np.array([f]).T, np.array([cv]).T
        # 离散化情况下，每个space的上界都不同
        upperbound=[]
        for sp in space_list:
            c_out=[]
            for n in space2names[sp]:
                supf=self.features_dict[n]
                c_out.append(int(supf.size(1))) # N,C,H,W
            c_out=list(set(c_out))
            assert len(c_out)<= 1,f'group{sp} channels not same:{c_out}'
            upperbound.append(c_out[0])
        # 上面是channels宽度的选择，接下来是各层metrics的选择
        gen_len=len(space2names)+len(self.features_dict)
        # MOD-start geatpy问题离散定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= gen_len,
            varTypes=[1]*gen_len,
            lb=[1]*len(space2names)+[0]*len(self.features_dict),
            ub=upperbound+[len(self.metric_options)]*len(self.features_dict),
            ubin=[0]*gen_len,
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='BG',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        self.logger.info(f"RESULTS are: {res}")
        # MOD-end
        
    def test_search_geatpy_discrete_metric_supp(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        assert self.metric, "This function needs denotion of searcher.metric"
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_= self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        # MOD-end
        assert self.metric in METRICS.keys(), \
            f"Metric {self.metric} not in M types:{METRICS.keys()}"

        vdict= self.features_dict if self.metric in METRICS.keys() else self.weights_dict
        filter_rank={k:METRICS[self.metric](v) for k,v in vdict.items()}

        # MOD-start evalVars
        def evalVars(Vars):
            f,cv=[],[]
            for Vs in Vars:
                # 计算 sim cka
                iszero,sum_sim=False,0
                assert len(Vs)==len(space_list)
                for space,cout_num in zip(space_list,Vs):
                    for n in space2names[space]:
                        supf=self.features_dict[n]
                        device=self.features_dict[n].device
                        ind =torch.topk(filter_rank[n],cout_num,largest=True)[1].to(device)
                        sind=torch.topk(filter_rank[n], supf.size(1)-cout_num, largest=False)[1].to(device)
                        supf=self.features_dict[n].index_select(1,sind) # 第1维，ind下标的通道(N,C,H,W)
                        subf=self.features_dict[n].index_select(1,ind) # 第1维，ind下标的通道(N,C,H,W)
                        supf=supf.view(supf.size(0),-1)
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f.append(sum_sim.data.cpu())
                # 计算FLOPS
                space2ratio={}
                for s,cn in zip(space_list,Vs):
                    n=space2names[s][0]
                    space2ratio[s]=cn/self.features_dict[n].size(1)
                sub_dict=pruner.sample_subnet_ratios(space2ratio)
                pruner.set_subnet(sub_dict)
                flops=self.algorithm.get_subnet_flops()
                rflops=self.algorithm.get_raw_flops()
                reduction_rate=(rflops-flops)/rflops
                cv.append(self.reduction_ratio- reduction_rate)

            return np.array([f]).T, np.array([cv]).T
        # 离散化情况下，每个space的上界都不同
        upperbound=[]
        for sp in space_list:
            c_out=[]
            for n in space2names[sp]:
                supf=self.features_dict[n]
                c_out.append(int(supf.size(1))) # N,C,H,W
            c_out=list(set(c_out))
            assert len(c_out)<= 1,f'group{sp} channels not same:{c_out}'
            upperbound.append(c_out[0])
        # MOD-start geatpy问题离散定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= len(space2names),
            varTypes=[1]*len(space2names),
            lb=[1]*len(space2names),
            ub=upperbound,
            ubin=[0]*len(space2names),
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='BG',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        self.logger.info(f"RESULTS are: {res}")
        # MOD-end
        

    def test_search_geatpy_discrete_inference(self):
        # MOD-start initial&log
        """Execute the pipeline of evolution search."""
        epoch_start = 0
        if self.resume_from is not None:
            searcher_resume = mmcv.fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['epoch'])
            self.logger.info('#' * 100)
            self.logger.info(f'Resume from epoch: {epoch_start}')
            self.logger.info('#' * 100)
        self.logger.info('Experiment setting:')
        self.logger.info(f'candidate_pool_size: {self.candidate_pool_size}')
        self.logger.info(f'candidate_top_k: {self.candidate_top_k}')
        self.logger.info(f'num_crossover: {self.num_crossover}')
        self.logger.info(f'num_mutation: {self.num_mutation}')
        self.logger.info(f'mutate_prob: {self.mutate_prob}')
        self.logger.info(f'max_epoch: {self.max_epoch}')
        self.logger.info(f'score_key: {self.score_key}')
        self.logger.info(f'constraints: {self.constraints}')
        self.logger.info('#' * 100)

        # self.features_dict={}
        space2names,self.features_dict,_=self.extract_features(self.algorithm_for_test,self.dataloader)
        supernet=self.algorithm.architecture
        pruner=self.algorithm.pruner
        name2space={n:k for k,v in space2names.items() for n in v}
        space_list=list(space2names.keys())
        # MOD-end
        # MOD-start evalVars
        def evalVars(Vars):
            f,cv=[],[]
            for Vs in Vars:
                # 计算FLOPS
                space2ratio={}
                for s,cn in zip(space_list,Vs):
                    n=space2names[s][0]
                    space2ratio[s]=cn/self.features_dict[n].size(1)+1e-6 #这里多加一个小数，为了数值稳定
                sub_dict=pruner.sample_subnet_ratios(space2ratio)
                pruner.set_subnet(sub_dict)
                flops=self.algorithm.get_subnet_flops()
                rflops=self.algorithm.get_raw_flops()
                reduction_rate=(rflops-flops)/rflops
                cv.append(self.reduction_ratio- reduction_rate)

                _, features_dict,_ =self.extract_features(self.algorithm_for_test,self.dataloader)
                # 计算 sim cka
                iszero,sum_sim=False,0
                assert len(Vs)==len(space_list)
                for space,cout_num in zip(space_list,Vs):
                    for n in space2names[space]:
                        supf=self.features_dict[n]
                        subf=features_dict[n]
                        supf=supf.view(supf.size(0),-1)
                        subf=subf.view(subf.size(0),-1)
                        sum_sim+=cka_t(gram_t(supf),gram_t(subf))
                f.append(sum_sim.data.cpu())

            return np.array([f]).T, np.array([cv]).T

        upperbound=[]
        for sp in space_list:
            c_out=[]
            for n in space2names[sp]:
                supf=self.features_dict[n]
                c_out.append(int(supf.size(1))) # N,C,H,W
            c_out=list(set(c_out))
            assert len(c_out)<= 1,f'group{sp} channels not same:{c_out}'
            upperbound.append(c_out[0])
        # MOD-start geatpy问题离散定义和求解部分
        problem=ea.Problem(
            name="search with cka",
            M=1,
            maxormins=[-1],
            Dim= len(space2names),
            varTypes=[1]*len(space2names),
            lb=[1]*len(space2names),
            ub=upperbound,
            ubin=[0]*len(space2names),
            evalVars=evalVars
        )
        solver=ea.soea_SGA_templet(
            problem=problem,
            population=ea.Population(Encoding='BG',NIND=self.candidate_pool_size),
            MAXGEN=self.max_epoch,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=5,
        )
        res=ea.optimize(
            seed=self.rand_seed,
            algorithm=solver,
            verbose=True,
            outputMsg=True,
            drawing=0,
            drawLog=False,
            saveFlag=False,
        )
        self.logger.info(f"RESULTS are: {res}")
        # MOD-end
        file_dict=[]
        for i,(cka,Vs) in enumerate(zip(res['lastPop'].ObjV, res['lasPop'].Chrom)):
            space2ratio={s:r for s,r in zip(space_list,Vs)}
            subnet_dict=pruner.sample_subnet_ratios(space2ratio)
            pruner.set_subnet(subnet_dict)
            chls=pruner.export_subnet()
            file_dict.append({
                'channel_cfg':chls,
                'cka':cka,
            })
        mmcv.dump(file_dict,"./test.json","json")

    def test_from_json(self):
        self.features_dict={}
        space2names,self.features_dict,_=self.extract_features(self.algorithm_for_test,self.dataloader)
        recipts_dicts=mmcv.load("./test.json","json")

        for i,recipt in enumerate(recipts_dicts):
            recipt=recipt['channel_cfg']
            sum_sim=0
            for n in self.features_dict:
                if not recipt.__contains__(n):continue
                supf=self.features_dict[n]
                k=int(recipt[n]['out_channels'])
                ind=torch.randperm(supf.size(1),device=supf.device)[:k]
                subf=self.features_dict[n].index_select(1,ind)

                supf=supf.view(supf.size(0),-1)
                subf=subf.view(subf.size(0),-1)
                sum_sim+=cka_t(gram_t(supf),gram_t(subf))
            print(f"{i}th model cka is :{sum_sim}")
    
