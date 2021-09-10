import torch.nn.functional as F


# Sparse layer
import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as ss
import math
import torch.nn.init as init


class SparseConv2D(torch.nn.Module):
    """Sparse 2d convolution.

    NOTE: Only supports NCHW format and no padding.
    """
    __constants__ = ['in_channels', 'out_channels']
    in_channels: int
    out_channels: int
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels,kernel_size,stride = 1,bias=True):
        super(SparseConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if (bias):
            self.bias = torch.nn.Parameter(torch.empty(out_channels,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_channels,self.in_channels*self.kernel_size*self.kernel_size, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        if (weight_np.ndim == 4):
            weight_np = weight_np.reshape(weight_np.shape[0],-1)
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        flat_x,out_shape = self.img2col(x)
        if not self.bias is None:
            flat_output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            flat_output = torch.sparse.mm(self.weight, flat_x)
        out =   flat_output.reshape([self.out_channels , *out_shape]).transpose(0,1)
        return out

    def img2col(self,x):
        # NCHW -> C*K*K, NHW
        input_windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # C,k*k,N, H, W
        input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1).permute(1,4,0,2,3)
        out_shape = [input_windows.shape[2],input_windows.shape[3],input_windows.shape[4]]
        input_windows = input_windows.reshape(input_windows.shape[0]*input_windows.shape[1],input_windows.shape[2]*input_windows.shape[3]*input_windows.shape[4]).contiguous()
        return input_windows,out_shape


class SparseConv1x1(torch.nn.Module):
    """Sparse 1x1 convolution.

    NOTE: Only supports 1x1 convolutions, NCHW format, unit
    stride, and no padding.
    """
    __constants__ = ['in_channels', 'out_channels']
    in_channels: int
    out_channels: int
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels,
            bias=True):
        super(SparseConv1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if (bias):
            self.bias = torch.nn.Parameter(torch.empty(out_channels,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_channels,self.in_channels, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
      # input NCHW
        input_shape = x.shape
        flat_x = x.transpose(0,1).reshape([input_shape[1],input_shape[0]*input_shape[2] * input_shape[3]]).contiguous()
        output_shape = [self.out_channels,input_shape[0], input_shape[2], input_shape[3]]
        if not self.bias is None:
            flat_output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            flat_output = torch.sparse.mm(self.weight, flat_x)
        out =   flat_output.reshape(output_shape).transpose(0,1)
        return out

class SparseLinear(torch.nn.Module):
    """Sparse linear layer.

    NOTE: (N ,*, H_in) format
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    def __init__(self,in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if (bias):
            self.bias = torch.nn.Parameter( torch.empty(out_features,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,sparsity = 0,weight_np = None ):
        if weight_np is None :
            weight_np = ss.random(self.out_features,self.in_features, density=1-sparsity).toarray()
        weight_np = weight_np.astype(np.float32)
        self.weight = torch.nn.Parameter(torch.from_numpy(weight_np).to_sparse().coalesce())
        self.build_bias()

    def build_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self,x):
        input_shape = list(x.shape)
        inner_shape = input_shape[1:-1]
        batch_size = input_shape[0]
        in_features = input_shape[-1]
        flat_x = x.transpose(0,-1).reshape([in_features, -1])
        if not self.bias is None:
            output = torch.sparse.addmm(self.bias,self.weight, flat_x)
        else:
            output = torch.sparse.mm(self.weight, flat_x)
        output = output.reshape([self.out_features, *inner_shape,batch_size]).transpose(0,-1)
        return output
    

def conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0,  bias=True):
  if (kernel_size == 1 and stride ==1  and padding == 0):
    return SparseConv1x1(in_channels, out_channels,bias)
  # elif padding == 0:
  #     return SparseConv2D(in_channels,out_channels,kernel_size,bias=bias,stride=stride)
  else:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)



####################################################################
######################       Resnet          #######################
####################################################################
#  def batch_mm(self, matrix_batch):
#     """
#     :param matrix: Sparse or dense matrix, size (m, n).
#     :param matrix_batch: Batched dense matrices, size (b, n, k).
#     :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
#     """
#     matrix = self.kernel
#     batch_size = matrix_batch.shape[0]
#     # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
#     vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

#     # A matrix-matrix product is a batched matrix-vector product of the columns.
#     # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
#     return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)

#   def img2col(input,k):
#     stride = 1
#     input_windows = input.unfold(2, k, stride).unfold(3, k, stride)
#     input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1).permute(0, 2,3,1,4)
#     input_windows = input_windows.reshape(input_windows.size()[0],input_windows.size()[1]*input_windows.size()[2],input_windows.size()[3]*input_windows.size()[4])
#     return input_windows.detach().cpu().numpy().copy()
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2D(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv2D(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        if stable_resnet:
            # Total number of blocks for Stable ResNet
            # https://arxiv.org/pdf/2002.08797.pdf
            L = 0
            for x in num_blocks:
                L+=x
            self.L = L
        else:
            self.L = 1
        
        self.masks = None

        self.conv1 = conv2D(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SparseLinear(in_planes*8*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.L))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out) / self.temp
        
        return out
            

def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model

def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model

def resnet50(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    return model

def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model

def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model

def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model
