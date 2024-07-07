import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import torch.nn.functional as F
import sys
import numpy as np
import torchvision
#from label_to_colormap import create_cityscapes_label_colormap
from sklearn.metrics import confusion_matrix
from PIL import Image

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

#def label2Color(label):
#    label = np.asarray(label, dtype=np.uint8)
#    colormap = create_cityscapes_label_colormap()
#    image = np.zeros((label.shape[0],label.shape[1],3), dtype=np.uint8)
#    for i in range(label.shape[0]):
#        for j in range(label.shape[1]):
#            if(label[i,j] > 19):
#                label[i,j] = 19
#            image[i,j] = colormap[label[i,j]]
#    return image

def segMap3(rgb, img_label, pred):
    # plotting for 0th batch only
    rgb, img_label, pred = rgb[0], img_label[0], pred[0]
    rgb = rgb.permute(1,2,0)

    pred = F.softmax(pred, dim=0)
    pred = torch.argmax(pred, dim=0)

    img_label = img_label.cpu()
    rgb = rgb.cpu()
    pred = pred.cpu()

    img_label = label2Color(img_label)
    pred = label2Color(pred)
    rgb = np.asarray(rgb, dtype=np.uint8)
    IMG_MEAN = np.array((104, 116, 122), dtype=np.uint8)
    rgb += IMG_MEAN
    rgb = rgb[:, :, : : -1]

    grid = torch.from_numpy(np.asarray([rgb, img_label, pred]))
    grid = grid.permute(0,3,1,2)
    grid = torchvision.utils.make_grid(grid)

    return grid


def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    udistr = torch.distributions.Uniform(0.5, 1.0)

    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    mu = udistr.sample((3, 1, 1)).numpy()
    #print('atrg', a_trg[:, h1:h2, w1:w2].min())
    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2] * mu
    #print('asrc', a_src[:, h1:h2, w1:w2].min())
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

##### AdaIN helper functions
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# only for single style image and single content image
def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def coral(cs, ct):
    d = sc.shape[0]
    loss = (cs - ct).pow(2).sum() / (4 * d **2)
    return loss

def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss

def calc_mu_sig(x, eps=1e-6):
    mu = x.mean(dim=[2,3], keepdim=True)
    var = x.var(dim=[2,3], keepdim=True)
    sig = (var + eps).sqrt()

    return mu, sig

def similarity(s_feat, prototypes):
    conf = [F.cosine_similarity(s_feat, prototype[..., None, None]) for prototype in prototypes]
    conf = torch.stack(conf, dim=1)
    return conf

def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

def compute_entropy(pred):
    output_sm = F.softmax(pred, dim=1).cpu().data[0].numpy().transpose(1,2,0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm + 1e-30)), axis=2, keepdims=False)
    output_ent = output_ent/np.log2(19)
    return output_ent

def process_label(label, num_classes):
    #device = torch.device("cuda")
    b, c, w, h = label.size()
    pred1 = torch.zeros(b, num_classes + 1, w, h).cuda()
    id = torch.where(label > -1, label, torch.Tensor([num_classes]).cuda())
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1

def calculate_mean_vector(feat, label, num_classes):
    scale_factor = F.adaptive_avg_pool2d(label, 1)
    vectors = []
    ids = []
    for n in range(feat.size()[0]):
        for t in range(num_classes):
            if scale_factor[n][t].item() == 0:
                continue
            #if (feat[n][t]>0).sum() < 10:
            #    continue
            s = feat[n] * label[n][t]
            s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
            vectors.append(s)
            ids.append(t)
    return vectors, ids

def update_vectors(prototype, prototype_num, id, vector, num_classes, momentum=0.9, name="moving_average", start_mean=False):
    if vector.sum().item() == 0:
        return
    if start_mean and prototype_num[id].item() < 100:
        name = "mean"
    if name == "moving_average":
        #print('moving average for prototypes')
        #print("prototype shape", prototype[id].shape)
        #print("vector shape", vector.shape)
        #print("momentum", momentum)
        prototype[id] = prototype[id] *  momentum + (1 - momentum) * vector.squeeze()
        prototype_num[id] += 1
        prototype_num[id] = min(prototype_num[id], 3000)
    elif name == "mean":
        #print("mean for prototypes")
        prototype[id] = prototype[id] * prototype_num[id] + vector.squeeze()
        prototype_num[id] += 1
        prototype[id] = prototype[id] / prototype_num[id]
        prototype_num[id] = min(prototype_num[id], 3000)
        pass
    else:
        raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


def feat_prototype_distance(prototype, feat, num_classes):
    N, C, H, W = feat.shape
    feat_proto_distance = -torch.ones((N, num_classes, H, W)).cuda()
    for i in range(num_classes):
        feat_proto_distance[:, i, :, :] =  torch.norm(prototype[i].reshape(-1, 1, 1).expand(-1, H, W)-feat, 2, dim=1)
    return feat_proto_distance

def get_prototype_weight(feat_proto_distance, proto_temperature):
    feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)
    feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance + 1e-6
    weight = F.softmax(-feat_proto_distance * proto_temperature, dim=1)
    return weight


def update_prototypes(pred, labels, prototype, prototype_num, num_classes, momentum=0.9):

    _, _, h, w = pred.size()    
    vectors = []
    ids = []

    interp_fea = nn.Upsample(size=(128, 256), mode='bilinear', align_corners=True).cuda()
    interp_lbl = nn.Upsample(size=(128, 256), mode='nearest', align_corners=None).cuda()

    pred = interp_fea(pred)
    label = interp_lbl(labels.unsqueeze(1).float())
    label = process_label(label, num_classes)
    scale_factor = F.adaptive_avg_pool2d(label, 1)

    # compute vectors and ids
    for n in range(pred.size()[0]):
        for t in range(num_classes):
            if scale_factor[n][t].item() == 0:
                continue
            #if (scale_factor[n][t] > 0).sum() < 10:
            #    continue
            s = pred[n] * label[n][t]
            s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
            vectors.append(s)
            ids.append(t)

            #proto[i] += torch.mean(pred4 * (labels==i), dim=[2,3])

        # update vectors and ids

        #print('ids', len(ids)) 
        for t in range(len(ids)):
            #print("update vectors")
            update_vectors(prototype, prototype_num, ids[t], vectors[t], num_classes, momentum, name='moving_average')

