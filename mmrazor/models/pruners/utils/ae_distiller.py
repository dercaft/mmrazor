# Copied from https://github.com/jamesoneill12/LayerFusion

import torch
from torch import nn
from torch.autograd import Variable

class AutoConv(nn.Module):
    def __init__(self,i_shape, o_shape) -> None:
        super(AutoConv,self).__init__()

        i_shape = (*i_shape[:2],i_shape[2]*i_shape[3])  # (C_in, C_out,W*W)
        o_shape = (*o_shape[:2],o_shape[2]*o_shape[3])
        # (C_in, C_out,W*W) -> (C_in_new, C_out, W*W) # C_out 看成是batchsize
        self.dconv1=nn.Conv1d(i_shape[0],o_shape[0],kernel_size=i_shape[-1],padding=i_shape[-1]//2)
        # (C_in_new, C_out, W*W) -> (C_in_new, C_out_new, W*W) # C_in_new 看成是batchsize
        self.dconv2=nn.Conv1d(i_shape[1],o_shape[1],kernel_size=i_shape[-1],padding=i_shape[-1]//2)
        # (C_in_new, C_out_new, W*W) -> (C_in_new, C_out_new, W*W) ww之间的信息交互
        self.dlinear=nn.Linear(i_shape[-1],o_shape[-1])

        self.elinear=nn.Linear(o_shape[-1],i_shape[-1])
        self.econv1=nn.Conv1d(o_shape[1],i_shape[1],kernel_size=o_shape[-1],padding=o_shape[-1]//2)
        self.econv2=nn.Conv1d(o_shape[0],i_shape[0],kernel_size=o_shape[-1],padding=o_shape[-1]//2)
        self.tanh=nn.Tanh()

    def forward(self,x):
        x=x.permute(1,0,2) # (C_in, C_out,W*W) -> (C_out, C_in,W*W)
        x=self.tanh(self.dconv1(x)) # (C_out, C_in,W*W) -> (C_out, C_in_new,W*W)
        x=x.permute(1,0,2) # (C_out, C_in_new,W*W) -> (C_in_new, C_out,W*W)
        x=self.tanh(self.dconv2(x)) # (C_in_new, C_out,W*W) -> (C_in_new, C_out_new,W*W)
        # reshape x into (-1,W*W)
        C_in_new,C_out_new,WW=x.shape
        x=x.reshape(C_in_new*C_out_new,WW)
        x=self.tanh(self.dlinear(x))
        x=self.tanh(self.elinear(x))
        x=x.reshape(C_in_new,C_out_new,WW)
        x=self.tanh(self.econv1(x)) # (C_in_new, C_out_new,W*W) -> (C_in_new, C_out,W*W)
        x=x.permute(1,0,2) # (C_in_new, C_out,W*W) -> (C_out, C_in_new,W*W)
        x=self.tanh(self.econv2(x)) # (C_out, C_in_new,W*W) -> (C_out, C_in,W*W)
        x=x.permute(1,0,2) # (C_out, C_in,W*W) -> (C_in, C_out,W*W)
        return x
    def get_embedding(self, x):
        x=x.permute(1,0,2) # (C_in, C_out,W*W) -> (C_out, C_in,W*W)
        x=self.tanh(self.dconv1(x)) # (C_out, C_in,W*W) -> (C_out, C_in_new,W*W)
        x=x.permute(1,0,2) # (C_out, C_in_new,W*W) -> (C_in_new, C_out,W*W)
        x=self.tanh(self.dconv2(x)) # (C_in_new, C_out,W*W) -> (C_in_new, C_out_new,W*W)
        # reshape x into (-1,W*W)
        C_in_new,C_out_new,WW=x.shape
        x=x.reshape(C_in_new*C_out_new,WW)
        x=self.tanh(self.dlinear(x))
        x=x.reshape(C_in_new,C_out_new,WW)
        return x

class Autoencoder(nn.Module):
    def __init__(self, x_dim, emb_dim, exp_decay=False):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, x_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

    def get_embedding(self, x):
        x = self.tanh(self.fc1(x))
        return x

    def forward_context(self, x, context):
        k = len(context)
        x = self.forward(x)
        context = [self.forward(cword) for cword in context]
        context = torch.exp(-k) * context
        return x, context



"""
 Performs a bilinear transformation of input pairs which have different size y = x1*A*x2 + b
 Shape:
    - x1 - N x d1
    - x2 - N x d2
    - A - (d1, d2, feature_size) 
"""

class SubMatrixAE(nn.Module):
    def __init__(self, xdim1, xdim2, emb_dim, exp_decay=False, bias=True, train_scheme='individual'):
        super(SubMatrixAE, self).__init__()

        self.f1 = nn.Bilinear(xdim1, xdim2, emb_dim)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x = self.tanh(self.f1(x1, x2))
        return (x)
    def get_embedding(self, x1, x2):
        x = self.tanh(self.f1(x1, x2))
        return x

def ae_distil(module, num_epochs=20):
    '''
    Args:
        module: the module to be distilled.
            module is added with in_mask and out_mask by Algorithm.
        num_epochs: the number of epochs to train the autoencoder.
    Return:
        module: the distilled module.
    '''
    assert hasattr(module,'in_mask') and hasattr(module,'out_mask'), \
        'module should be added with in_mask and out_mask by Algorithm.'
    old_parameters={}
    # print("in_mask shape ",module.in_mask.shape)
    # print("out_mask shape ",module.out_mask.shape)
    in_dim = int(module.in_mask.sum())
    out_dim = int(module.out_mask.sum())
    if in_dim == 0 or out_dim == 0 or in_dim == module.in_mask.size(0) or out_dim == module.out_mask.size(0):
        return module
    def ae(parameter,new_shape):
        num = 1
        for i in new_shape:
            num *= i
        criterion = torch.nn.MSELoss()
        print("parameter ",parameter.shape," new_shape ",new_shape," num ",num)
        if len(parameter.shape)<4:
            old_parameter=torch.reshape(parameter,(-1,))
            model = Autoencoder(int(old_parameter.size(0)), int(num))
            model=model.to('cuda')
            old_parameter=old_parameter.to('cuda')
            opt = torch.optim.Adam(model.parameters())
            loss=0
            for i in range(num_epochs):
                opt.zero_grad()
                p_hat = model(old_parameter)
                loss = criterion(p_hat, old_parameter)
                loss.backward()
                opt.step()
            print("FINAL loss: ", loss)
            p_emb = model.get_embedding(old_parameter)
            p_emb = p_emb.reshape(new_shape)
        else:
            old_parameter=torch.reshape(parameter,(*parameter.shape[:2],-1))
            model = AutoConv(parameter.shape,new_shape)
            model=model.to('cuda')
            old_parameter=old_parameter.to('cuda')
            opt = torch.optim.Adam(model.parameters())
            loss=0
            for i in range(num_epochs):
                opt.zero_grad()
                p_hat = model(old_parameter)
                loss = criterion(p_hat, old_parameter)
                loss.backward()
                opt.step()
            print("FINAL loss: ", loss)
            p_emb = model.get_embedding(old_parameter)
            p_emb = p_emb.reshape(new_shape)        
        return p_emb
    if hasattr(module, 'bias') and module.bias is not None:
        old_parameters['bias'] = module.bias.data.clone()
        new_shape=(out_dim,*module.bias.shape[1:])
        p_emb=ae(module.bias,new_shape)
        module.bias.data[:out_dim] = p_emb
    if hasattr(module, 'weight') and module.weight is not None:
        old_parameters['weight'] = module.weight.data.clone()
        new_shape=(out_dim,in_dim,*module.weight.shape[2:])
        p_emb=ae(module.weight,new_shape)
        module.weight.data[:out_dim,:in_dim] = p_emb
    old_parameters['in_mask'] = module.in_mask.clone()
    old_parameters['out_mask'] = module.out_mask.clone()
    module.in_mask = torch.zeros_like(module.in_mask)
    module.in_mask[:,:in_dim,:,:] = 1
    module.out_mask = torch.zeros_like(module.out_mask)
    module.out_mask[:,:out_dim,:,:] = 1

    return module, old_parameters
