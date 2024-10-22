# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class DeeplabResnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(DeeplabResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.mixstyle = MixStyle()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, style=None):
        input_size = x.size()[2:]
        if style != None:
            style0, style1, style2, style3, style4 = style
            x = self.mixstyle(x, style0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        if style != None:
            feat3 = self.mixstyle(feat3, style3)

        output1 = self.layer5(feat3)
        ouptut1 = F.interpolate(output1, size=input_size, mode='bilinear', align_corners=True)

        feat4 = self.layer4(feat3)
        if style != None:
            feat4 = self.mixstyle(feat4, style4)
        output2 = self.layer6(feat4)
        output2 = F.interpolate(output2, size=input_size, mode='bilinear', align_corners=True)

        return output2, (x, feat1, feat2, feat3, feat4) # changed!

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]


# def DeeplabResNet(num_classes=21, pretrained=True):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
#
#     if pretrained:
#         restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
#         saved_state_dict = torch.load(restore_from)
#
#         new_params = model.state_dict().copy()
#         for i in saved_state_dict:
#             i_parts = i.split('.')
#             if not i_parts[1] == 'layer5':
#                 new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
#         model.load_state_dict(new_params)
#     return model




class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, restore_from=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        # vgg = VGG
        if pretrained:
            vgg.load_state_dict(torch.load(restore_from))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': args.lr}]  #########


# reference: http://github.com/KaiyangZhou/mixstyle-release
class MixStyle(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.udistr = torch.distributions.Uniform(0.0, 1.0)
        self.eps = eps

    def forward(self, x, style):
        B = x.size(0)
        C = x.size(1)

        mu = x.mean(dim=[2,3], keepdim=True)
        var = x.var(dim=[2,3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu2 = style.mean(dim=[2,3], keepdim=True)
        var2 = style.var(dim=[2,3], keepdim=True)
        sig2 = (var2 + self.eps).sqrt()
        
        mu2_gau = torch.randn(mu2.shape).cuda()*0.1*(mu2-mu)
        sig2_gau = torch.randn(sig2.shape).cuda()*0.1*(sig2-sig) 
        mu2_p = mu2_gau + mu2
        sig2_p = sig2_gau + sig2

        udistr = self.udistr.sample((B,C,1,1))
        udistr = udistr.to(x.device)
        mu_mix = mu*udistr + mu2_p * (1-udistr)
        sig_mix = sig * udistr + sig2_p * (1-udistr)

        return x_normed * sig_mix + mu_mix
