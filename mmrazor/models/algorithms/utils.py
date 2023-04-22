import torch
import math

from mmrazor.models.pruners.metrics import feature_pool
def generate_hook(name,fdict:dict,fpool:dict|None=None,device:str='cuda:0'):
    def output_hook(module,input,output) -> None:
        # if name exists in fdict, it is a tensor[N,C,H,W], add new tensor on axis of N
        if name in fdict:
            fdict[name]=torch.cat([fdict[name],output.detach().to(device)],dim=0)
        else:
            fdict[name]=output.detach().to(device)
        if isinstance(fpool,dict) :
            if name in fpool:
                fpool[name]=torch.cat([fpool[name],feature_pool(output.detach().to(device))],dim=0)
            else:
                fpool[name]=feature_pool(output.detach().to(device))
    return output_hook

def extract_features(inference,supernet,pruner, dataloader, get_pool:bool=False, rate:float=0.0, device:str='cuda:0'):
    """
        Use hooks to get features from model's forward.
        Return: space2name: {k:[n1,...,nn]}
    """
    assert rate>=0 and rate<=1
    # supernet=self.architecture if hasattr(self,'architecture') else self.algorithm.architecture
    # pruner=self.pruner if hasattr(self,'pruner') else self.algorithm.pruner

    space2name={}
    for name,module in supernet.model.named_modules():
        spi=pruner.name2space.get(name,None)
        if not isinstance(spi,str): continue
        g=space2name.setdefault(spi,[])
        g.append(name)
        space2name[spi]=g
    
    hookers=[]
    features_dict={}
    features_pool={} if get_pool else None
    names=list(pruner.name2space.keys())
    for name, module in supernet.model.named_modules():
        if name in names:
            oo=generate_hook(name,features_dict,features_pool,device=device)
            hooker=module.register_forward_hook(oo)
            hookers.append(hooker)
            # self.weights_dict[name]=module.weight
            # if hasattr(module,'bias'):
                # self.bias_dict[name]=module.bias
    
    # forward a batch
    # algorithm_for_test.
    # set supernet, data to cuda
    # dataloader.cuda()
    # supernet.cuda()
    # device: str = 'cuda:0'
    # torch.Tensor().device
    # if device != 'cpu':
    #     supernet = MMDataParallel(
    #         supernet.to(device), device_ids=[0])
    bn=math.ceil(len(dataloader)*rate) if rate>0 else 1
    inference.eval()
    for i, data_batch in enumerate(dataloader):
        if i>bn:break
        inference.forward(**data_batch,return_loss=False)
        # supernet.val_step(**data_batch,None,return_loss=False)
    for hooker in hookers:
        hooker.remove()

    return space2name,features_dict,features_pool

