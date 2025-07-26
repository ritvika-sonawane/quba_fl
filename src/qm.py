import torch 
from torch import nn
import torch.nn.functional as F
from src.ops import StoShift

def combine_scale(s1, s2):
    return [s1[0] * s2[0]]

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, quantizer, weight_update, 
                 initialize='uniform',  bias=False):
        super(QLinear, self).__init__()

        # self.layer_init(in_features, out_features, qmode)
        self.quantizer = quantizer
        if bias:
            self.weight = torch.zeros([out_features, in_features+1])
        else:
            self.weight = torch.zeros([out_features, in_features])
        if initialize == 'uniform':
            torch.nn.init.xavier_uniform_(self.weight)
        elif initialize == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        
        self.weight, self.weight_scale= self.quantizer(self.weight)
        self.weight_update = weight_update
        self.bias = bias

    def forward(self, input):
        """ save activation for backwards """
        act, act_s = input

        if self.bias:
            act = torch.cat((act, torch.ones(act.shape[0], 1)), dim=1)
        self.act_in = act, act_s 

        out = torch.matmul(act, self.weight.T)

        out_s = combine_scale(act_s, self.weight_scale)
        return out, out_s
    

    def backward(self, input):
        # err: B x out_features
        err, err_s = input
        act, act_s = self.act_in

        self.grad = torch.matmul(err.T, act)
        self.grad_scale = combine_scale(err_s, act_s)

        out = torch.matmul(err, self.weight)
        if self.bias:
            out = out[:, :-1]
        out_s = combine_scale(err_s, self.weight_scale)

        self.weight = self.weight_update(self.weight, self.weight_scale, self.grad, self.grad_scale)

        return out, out_s



class QConv2d(nn.Module):
    '''
    conv 3x3 with dilation 1, stride 1, padding 1
    NHWC format
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 quantizer, weight_update, initialize='uniform'):
        super(QConv2d, self).__init__()
        self.quantizer = quantizer

        self.weight = torch.zeros([out_channels, in_channels, kernel_size, kernel_size])
        if initialize == 'uniform':
            torch.nn.init.xavier_uniform_(self.weight)
        elif initialize == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        
        self.weight, self.weight_scale = self.quantizer(self.weight)
        self.weight_update = weight_update
        self.bias = False
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        act, act_s = input
        self.act_in = act, act_s

        out = F.conv2d(act, self.weight, stride=self.stride, padding=self.padding)
        out_s = combine_scale(act_s, self.weight_scale)
        return out, out_s

    def backward(self, input):
        err, err_s = input

        act, act_s = self.act_in

        out = torch.nn.grad.conv2d_input(act.shape, self.weight, 
                                         err, self.stride, self.padding)
        self.grad = torch.nn.grad.conv2d_weight(act, self.weight.shape, err, self.stride, self.padding) 
        self.grad_scale = combine_scale(err_s, act_s)
        out_s = combine_scale(err_s, self.weight_scale)

        self.weight = self.weight_update(self.weight, self.weight_scale, self.grad, self.grad_scale)
        return out, out_s
        

class QReLU(nn.Module):
    def __init__(self, forward_shift, backward_shift) -> None:
        super().__init__()
        self.forward_shift = forward_shift
        self.backward_shift = backward_shift


    def forward(self, input):
        self.act_in = input
        act, act_s = input
        out = torch.max(act, torch.tensor(0))
        out, out_s = self.forward_shift(out, act_s)
        return out, out_s

    def backward(self, input):
        err, err_s = input
        err, err_s = self.backward_shift(err, err_s)
        act, _ = self.act_in
        out = torch.where(act>0, err, torch.tensor(0))
        out_s = err_s
        return out, out_s

class QMaxpool2d(nn.Module):
    '''
    Integer Max Pooling 2d Layer
    '''
    def __init__(self, kernel_size, stride, padding=0):
        super(QMaxpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,input):
        act, act_s = input
        out, self.indices = F.max_pool2d(act, self.kernel_size, self.stride, self.padding, return_indices=True)
        out_s = act_s
        return out, out_s

    def backward(self, input):
        """ cudnn pool backward function doesn't support int8, use half instead"""
        err, err_s = input
        out = F.max_unpool2d(err.float(), self.indices, self.kernel_size, self.stride, self.padding)
        return out, err_s

class QFlat(nn.Module):
    ''' Flat the input integer tensor except the batch dimension '''
    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        return act_in.view(act_in.size(0), -1), exp_in

    def backward(self,input):
        '''
        Convert the Flat error back to the shape before flattern
        '''
        err_in, err_exp = input
        act, _ = self.act_in
        return err_in.view_as(act), err_exp


    
class QDropout(nn.Module):
    '''
    Integer Dropout layer
    '''
    def __init__(self, p=0.5,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.training = True

    def forward(self, input):
        if self.training:
            act_in, exp_in = input
            self.drop_mask = torch.randint(low=0, high=2, size=(act_in.size(1),))
            self.drop_mask = torch.where(self.drop_mask == 0, torch.tensor(0), torch.tensor(1))
            return act_in*self.drop_mask, [exp_in[0] * 2]
        return input

    def backward(self, input):
        err_in, err_exp = input
        err_out = err_in*self.drop_mask
        return err_out, err_exp


class QCELoss(nn.Module):
    def __init__(self, quantizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer

    def forward(self, out_val, scale, target):
        x =  F.softmax(out_val * scale[0], dim=-1) - F.one_hot(target, out_val.size(1))
        return self.quantizer(x)   

class QMSELoss(nn.Module):
    def __init__(self, quantizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer

    def forward(self, out_val, scale, target):
        x =  (out_val * scale[0] - target) * 2
        return self.quantizer(x)  

