import torch
import math
from mmrazor.models.pruners.metrics import feature_pool
def generate_hook(name,fdict:dict,fpool:dict|None=None):
    def output_hook(module,input,output) -> None:
        # if name exists in fdict, it is a tensor[N,C,H,W], add new tensor on axis of N
        if name in fdict:
            fdict[name]=torch.cat([fdict[name],output.detach()],dim=0)
        else:
            fdict[name]=output.detach().cpu()
        if isinstance(fpool,dict) :
            if name in fpool:
                fpool[name]=torch.cat([fpool[name],feature_pool(output.detach())],dim=0)
            else:
                fpool[name]=feature_pool(output.detach().cpu())
    return output_hook

def extract_features(self, dataloader, get_pool:bool=False, rate:float=0.0):
    """
        Use hooks to get features from model's forward.
        Return: space2name: {k:[n1,...,nn]}
    """
    assert rate>=0 and rate<=1
    supernet=self.architecture
    pruner=self.pruner

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
            oo=generate_hook(name,features_dict,features_pool)
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
    
    bn=math.ceil(len(dataloader)*rate) if rate>0 else 1
    supernet.eval()
    for i, data_batch in enumerate(dataloader):
        if i>bn:break
        # data_batch['img'].cuda()
        # print(data_batch.keys())
        supernet.forward(**data_batch,return_loss=False)
    for hooker in hookers:
        hooker.remove()

    return space2name,features_dict,features_pool

