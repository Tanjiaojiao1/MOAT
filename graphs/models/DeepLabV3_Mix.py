import sys
import os
import numpy as np
sys.path.append(os.path.abspath('.'))
from graphs.models.deeplab_multi import *
from .utils import *

img_mean = np.asarray((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# reference: http://github.com/KaiyangZhou/mixstyle-release
class MixStyle(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.udistr = torch.distributions.Uniform(0.5, 1.0)
        self.gdistr = torch.distributions.Normal(0, 1.0)
        self.eps = eps

    def forward(self, x, style):
        B = x.size(0)
        C = x.size(1)

        mu = x.mean(dim=[2,3], keepdim=True)
        var = x.var(dim=[2,3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x -mu) / sig

        mu2 = style.mean(dim=[2,3], keepdim=True)
        var2 = style.var(dim=[2,3], keepdim=True)
        sig2 = (var2 + self.eps).sqrt()

        mu2_gau = torch.randn(mu2.shape).cuda()* 0.5 *(mu2-mu)
        sig2_gau = torch.randn(sig2.shape).cuda()* 0.5 *(sig2-sig)
        mu2_p = mu2_gau + mu2
        sig2_p = sig2_gau + sig2

        udistr = self.udistr.sample((B, C, 1, 1))
        udistr = udistr.to(x.device)
        mu_mix = mu * udistr + mu2_p * (1-udistr)
        sig_mix = sig * udistr + sig2_p * (1-udistr)

        #sig_mix, sig_mix = mu2_p, sig2_p

        return x_normed * sig_mix + mu_mix


class UncertaintyMix(nn.Module):
    def __init__(self, eps=1e-6, p=0.7, dim=-1):
        super().__init__()
        self.udistr = torch.distributions.Beta(0.5, 0.5)
        self.gdistr = torch.distributions.Normal(0.0, 1.0)
        self.eps = eps
        self.p = p
        self.dim = dim
        self.factor=1.0

    def forward(self, x, style):
        B = x.size(0)
        C = x.size(1)

        mu = x.mean(dim=[2,3], keepdim=True)
        var = x.var(dim=[2,3], keepdim=True)
        sig = (var + self.eps).sqrt()
        #mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) /sig

        mu2 = style.mean(dim=[2,3], keepdim=True)
        var2 = style.var(dim=[2,3], keepdim=True)
        sig2 = (var2 + self.eps).sqrt()

        #mu2_std = ((mu2-mu).var(dim=0, keepdim=True) + self.eps).sqrt()
        #sig2_std = ((sig2-mu).var(dim=0, keepdim=True) + self.eps).sqrt()

        mu2_std = ((mu2-mu) + self.eps)
        sig2_std = ((sig2-sig) + self.eps)

        udistr = self.udistr.sample(mu2_std.shape).cuda()
        mu2_p = udistr * self.factor * mu2_std + mu2
        sig2_p = udistr * self.factor * sig2_std + sig2
       
        #mu2_p = torch.randn(mu2_std.shape).cuda() * self.factor * mu2_std + mu2
        #sig2_p = torch.randn(sig2_std.shape).cuda() * self.factor * sig2_std + sig2


        #mu2_gau = torch.randn(mu2.shape).cuda()*0.1*(mu2-mu_s)
        #sig2_gau = torch.randn(sig2.shape).cuda()*0.1*(sig2-sig_s)
        #mu2_p = mu2_gau + mu2
        #sig2_p = sig2_gau +sig2

        udistr = self.udistr.sample((B, C, 1, 1))
        udistr = udistr.to(x.device)
        mu_mix = mu * udistr + mu2_p * (1-udistr)
        sig_mix = sig * udistr + sig2_p * ( 1-udistr)

        if np.random.random() > self.p:
            mu_mix, sig_mix = mu, sig
        else:
            mu_mix, sig_mix = mu2_p, sig2_p

        return x_normed * sig_mix + mu_mix

class DeeplabResnetMix(ResNetMulti):

    def __init__(self, block, layers, num_classes):
        super(DeeplabResnetMix, self).__init__(block, layers, num_classes)
  
        #self.mixstyle = MixStyle()
        self.mixstyle = UncertaintyMix()

    def forward(self, img, style=None):
        input_size = img.size()[2:]
        if style != None:
            style0, style1, style2, style3, style4 = style
            img_npy = img.cpu().numpy().squeeze(0).transpose(1,2,0)
            img_npy = img_npy.transpose(2,0,1)
            img_npy = toimage(img_npy, channel_axis=0, cmin=0, cmax=255)
            img_npy.save('img_org.png')
            img = self.mixstyle(img, style0)
            img_npy = img.cpu().numpy().squeeze(0).transpose(1,2,0)
            img_npy = img_npy.transpose(2,0,1)
            img_npy = toimage(img_npy, channel_axis=0, cmin=0, cmax=255)
            img_npy.save('img_mix.png')
            #print("using mixstyle")
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        if style != None:
            feat1 = self.mixstyle(feat1, style1)
        feat2 = self.layer2(feat1)
        if style != None:
            feat2 = self.mixstyle(feat2, style2)
        feat3 = self.layer3(feat2)
        if style != None:
            feat3 = self.mixstyle(feat3, style3)

        feat4 = self.layer4(feat3)
        if style != None:
            feat4 = self.mixstyle(feat4, style4)
        #feat = x
        output, res = self.layer6(feat4)  # classifier module
              
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)


        return output, res, (img, feat1, feat2, feat3, feat4)


"""
class DeeplabVGGFeat(DeeplabVGG):

    def forward(self, x):

        input_size = x.size()[2:]
        x = self.features(x)
        feat = x
        x1 = self.classifier(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        return x1, feat
"""

def DeeplabMix(num_classes, backbone, pretrained=True):
    print('DeeplabV2 is being used with {} as backbone'.format(backbone))
    if backbone == 'ResNet101':
        model = DeeplabResnetMix(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)
        if pretrained:
            restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
    #elif backbone == 'VGG16':
    #    restore_from = './pretrained_model/vgg16-397923af.pth'
    #    model = DeeplabVGGFeat(num_classes, restore_from=restore_from, pretrained=pretrained)
    else:
        raise Exception

    return model





