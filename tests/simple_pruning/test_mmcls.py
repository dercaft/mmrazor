# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger

from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, save_checkpoint

from mmrazor.models.builder import build_algorithm
from mmrazor.utils import setup_multi_processes

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    # Adder
    parser.add_argument('--reduction_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--checkpoint', default='', help='checkpoint file of algorithm')
    parser.add_argument('--checkpoint_model', default='', help='checkpoint file of model')
    parser.add_argument('--execute_function',type=str,default='', help='name of search function')
    parser.add_argument('--save_compress_algo', default='', help='save compress algorithm')
    #
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    logger=get_root_logger(log_level='INFO')
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the algorithm and load checkpoint
    # Show configs in algorithm
    for k, v in cfg.algorithm.items():
        logger.info(f'INFO: {k}: {v}')

    algorithm = build_algorithm(cfg.algorithm)
    model = algorithm.architecture.model
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # Adder
    if args.checkpoint:
        checkpoint = load_checkpoint(
            algorithm, args.checkpoint, map_location='cpu')
    elif args.checkpoint_model:
        checkpoint = load_checkpoint(
            algorithm.architecture.model, args.checkpoint_model, map_location='cpu')
    else:
        warnings.warn('No Checkpoint loaded!')
    logger.info(f'INFO: checkpoints contains keys: {checkpoint.keys()}')
    #
    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    # Adder Create arguments dict
    args_dict = {
        'dataloader': data_loader,
        'reduction_ratio':args.reduction_ratio,
        'save_compress_algo':args.save_compress_algo,
    }
    # algo_device = algorithm.device
    # # # test MODEL 
    # if not distributed:
    #     if args.device == 'cpu':
    #         algorithm = algorithm.cpu()
    #     else:
    #         algorithm = MMDataParallel(algorithm, device_ids=cfg.gpu_ids)
    #     model.CLASSES = CLASSES
    #     show_kwargs = {} if args.show_options is None else args.show_options
    #     outputs = single_gpu_test(algorithm, data_loader, args.show,
    #                               args.show_dir, **show_kwargs)
    # else:
    #     algorithm = MMDistributedDataParallel(
    #         algorithm.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(algorithm, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     results = {}
    #     if args.metrics:
    #         eval_results = dataset.evaluate(outputs, args.metrics,
    #                                         args.metric_options)
    #         results.update(eval_results)
    #         for k, v in eval_results.items():
    #             logger.info(f'\n{k} : {v:.2f}')
    
    # if hasattr(algorithm, 'module'):
    #     algorithm = algorithm.module
    # else:
    #     algorithm = algorithm
    # algorithm = algorithm.cpu()
    # Adder 多进程对压缩操作的影响：
    rank, _ = get_dist_info()
    # 如果每个进程的模型不同，那么应该会出现不同的进程，其sub model的flop不同
    if rank==0:
        logger.info(f"RANK {rank} DO compress")
        execute_function=getattr(algorithm,args.execute_function,None)
        assert execute_function,"WARNING: Execute function is None"
        execute_function(**args_dict)
        # algorithm.autoencoder_new_weight()
    else:
        logger.info(f"RANK {rank} NOT compress")
    rflop=algorithm.get_supnet_flops()
    sflop=algorithm.get_subnet_flops()
    logger.info(f"RANK {rank} Raw model flop is: {rflop}")
    logger.info(f"RANK {rank} Sub model flop is: {sflop}")
    # # test MODEL 
    # if not distributed:
    #     if args.device == 'cpu':
    #         algorithm = algorithm.cpu()
    #     else:
    #         algorithm = MMDataParallel(algorithm, device_ids=cfg.gpu_ids)
    #     model.CLASSES = CLASSES
    #     show_kwargs = {} if args.show_options is None else args.show_options
    #     outputs = single_gpu_test(algorithm, data_loader, args.show,
    #                               args.show_dir, **show_kwargs)
    # else:
    #     algorithm = MMDistributedDataParallel(
    #         algorithm.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(algorithm, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     results = {}
    #     if args.metrics:
    #         eval_results = dataset.evaluate(outputs, args.metrics,
    #                                         args.metric_options)
    #         results.update(eval_results)
    #         for k, v in eval_results.items():
    #             logger.info(f'\n{k} : {v:.2f}')
    #     if args.out:
    #         scores = np.vstack(outputs)
    #         pred_score = np.max(scores, axis=1)
    #         pred_label = np.argmax(scores, axis=1)
    #         pred_class = [CLASSES[lb] for lb in pred_label]
    #         results.update({
    #             'class_scores': scores,
    #             'pred_score': pred_score,
    #             'pred_label': pred_label,
    #             'pred_class': pred_class
    #         })
    #         logger.info(f'\ndumping results to {args.out}')
    #         mmcv.dump(results, args.out)
    # if len(args.save_compress_algo):
    #     logger.info(f"RANK {rank} Save compress model to {args.save_compress_algo}")
    #     save_checkpoint(algorithm, args.save_compress_algo, meta={'CLASSES': CLASSES})

if __name__ == '__main__':
    main()
