
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from Layers import layers

####################################################################
######################       Resnet          #######################
####################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layers.BatchNorm2d(planes)
        self.conv2 = layers.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layers.BatchNorm2d(planes)
        
        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                layers.BatchNorm2d(self.expansion*planes)
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
        self.conv1 = layers.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = layers.BatchNorm2d(planes)
        self.conv2 = layers.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = layers.BatchNorm2d(planes)
        self.conv3 = layers.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = layers.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                layers.BatchNorm2d(self.expansion*planes)
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

        self.conv1 = layers.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layers.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layers.Linear(in_planes*8*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        all_layers = []
        for stride in strides:
            all_layers.append(block(self.in_planes, planes, stride, self.L))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*all_layers)

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


####################################################################
#######################   VGG    ###################################
####################################################################

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='CIFAR10', depth=19, cfg=None, affine=True, batchnorm=True,
                 init_weights=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset.upper() == 'CIFAR10':
            num_classes = 10
        elif dataset.upper() == 'CIFAR100':
            num_classes = 100
        elif dataset.upper() == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = layers.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init)


    def make_layers(self, cfg, batch_norm=False):
        all_layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                all_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layers.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    all_layers += [conv2d, layers.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    all_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

def vgg19(dataset):
    return VGG(dataset=dataset,depth=19)

def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()