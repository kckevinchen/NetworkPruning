from scipy import sparse
import torch.nn as nn
import torch
import numpy as np
import os, sys
import scipy.io as sio
from scipy.sparse import coo_matrix
from scipy.linalg import toeplitz

from experiments.sparse_resnet import SparseConv1x1, SparseConv2D,SparseLinear

# from IPython import embed


def toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))) ) 

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv
def conv2d_toeplitz(kernel, input):
    """Compute 2d convolution over multiple channels via toeplitz matrix
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input: shape=(n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    input_size = input.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    output = np.zeros(output_size)

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            output[i] += T_k.dot(input[j].flatten()).reshape(output_size[1:])  # sum over input channels

    return output


def load_from_mtx(net, mtx_dir):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    
    If apply_hooks == False, then set weight to 0 but do not block the gradient.
    This is used for FORCE algorithm that sparsifies the net instead of pruning.
    """
    
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all non-prunable modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())
    mtx_files = sorted(os.listdir( mtx_dir ), key= lambda x: int(x.split("_")[0]))
    keep_masks = []
    hook_handlers = []

    for layer, mtx_f in zip(prunable_layers, mtx_files):

        print(layer)
        print(mtx_f)
        matrix = np.array(sio.mmread(os.path.join(mtx_dir,mtx_f)).todense()).T
        print(matrix.shape)
        print(layer.weight.data.shape)
        if isinstance(layer, nn.Conv2d):
             matrix =  matrix.reshape(layer.weight.data.shape)
        tensor = torch.from_numpy(matrix)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """
            def hook(grads):
                return grads * keep_mask

            return hook

        keep_mask = (tensor!= 0.).float().cuda()
        layer.weight.data = tensor.float().cuda()
        hook_handlers.append(layer.weight.register_hook(hook_factory(keep_mask)))

        keep_masks.append(keep_mask)

    return hook_handlers, keep_masks

    

def save_to_mtx(net, mtx_dir):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    
    If apply_hooks == False, then set weight to 0 but do not block the gradient.
    This is used for FORCE algorithm that sparsifies the net instead of pruning.
    """
    
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all non-prunable modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())


    i = 0
    for layer in prunable_layers:
        if isinstance(layer, nn.Conv2d):
            name = "{}_conv2d".format(i)
            raw_weight = layer.weight.data.cpu().numpy()
            compressed_weight = raw_weight.reshape(raw_weight.shape[0],-1).T
            sparse_matrix = coo_matrix(compressed_weight)
        else:
            name = "{}_linear".format(i)
            sparse_matrix = coo_matrix(layer.weight.data.cpu().numpy())

        sio.mmwrite(os.path.join(mtx_dir,name),sparse_matrix)
        i += 1
def save_matrix(m,m_dir):
    sparse_matrix = coo_matrix(m)
    sio.mmwrite(m_dir,sparse_matrix)

def reorder_dir(dir,input=True):
    files = [f for f in os.listdir(dir)]
    files.sort(key= lambda x: x.split("_")[-1])
    i = 0
    if(input):
        for j in range(len(files)//2):
            t = files[2*j].split("_")[-1].split(".")[0]
            if(files[2*j].split("_")[-1]!= files[2*j+1].split("_")[-1]):
                print("error")
            else:
                file_1 = os.path.join(dir,files[2*j])
                file_2 = os.path.join(dir,files[2*j+1])
                os.rename(file_1, file_1.replace(t,str(i)))
                os.rename(file_2, file_2.replace(t,str(i)))
                i += 1
    else:
        for j in range(len(files)):
            t = files[j].split("_")[-1].split(".")[0]
            file = os.path.join(dir,files[j])
            os.rename(file, file.replace(t,str(i)))
            i += 1

def sparse_load_from_mtx(sparse_net, mtx_dir,device):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, SparseConv1x1) or isinstance(layer,SparseLinear) or isinstance(layer,SparseConv2D) , sparse_net.modules())

    mtx_files = sorted(os.listdir( mtx_dir ), key= lambda x: int(x.split("_")[0]))
    keep_masks = []
    hook_handlers = []

    for layer, mtx_f in zip(prunable_layers, mtx_files):
        matrix = np.array(sio.mmread(os.path.join(mtx_dir,mtx_f)).todense())
        print(layer)
        if isinstance(layer, nn.Conv2d):
            matrix =  matrix.reshape(layer.weight.data.shape)
            tensor = torch.from_numpy(matrix)

            def hook_factory(keep_mask):
                """
                The hook function can't be defined directly here because of Python's
                late binding which would result in all hooks getting the very last
                mask! Getting it through another function forces early binding.
                """
                def hook(grads):
                    return grads * keep_mask

                return hook

            keep_mask = (tensor!= 0.).float().to(device)
            layer.weight.data = tensor.float().to(device)
            hook_handlers.append(layer.weight.register_hook(hook_factory(keep_mask)))

            keep_masks.append(keep_mask)
            
        elif isinstance(layer,SparseConv1x1) or isinstance(layer,SparseLinear) or isinstance(layer,SparseConv2D):
            matrix =  matrix.reshape(layer.weight.data.shape)
            print(np.count_nonzero(matrix)/np.prod(matrix.shape))
            layer.reset_parameters(weight_np=matrix)
    return hook_handlers, keep_masks

if __name__ == '__main__':
    reorder_dir("./mtx")
