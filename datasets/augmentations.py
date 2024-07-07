# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

from PIL import Image
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np

random_mirror = True

def Identity(img, label, v):
    return img, label

def ShearX(img, label, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    image = img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
    return image, label


def ShearY(img, label, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    image = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    return image, label


def TranslateX(img, label, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    image = img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    return image, label


def TranslateY(img, label, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    image = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    return image, label


def TranslateXAbs(img, label, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    image = img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    return image, label


def TranslateYAbs(img, label, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    image = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    label = label.transform(label.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    return image, label


def Rotate(img, label, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    image = img.rotate(v)
    label = label.rotate(v)
    return image, label


def AutoContrast(img, label, _):
    image = PIL.ImageOps.autocontrast(img)
    return image, label


def Invert(img, label, _):
    image = PIL.ImageOps.invert(img)
    return image, label


def Equalize(img, label, _):
    image = PIL.ImageOps.equalize(img)
    return image, label


def Flip(img, label, _):  # not from the paper
    image = PIL.ImageOps.mirror(img)
    label = PIL.ImageOps.mirror(label)
    return image, label


def Solarize(img, label, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), label


def Posterize(img, label, v):  # [4, 8]
    assert 0 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v), label


def Posterize2(img, label, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v), label


def Contrast(img, label, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), label


def Color(img, label, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), label


def Brightness(img, label, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v), label


def Sharpness(img, label, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), label

def Hue(img, label, v): # [-0.5, 0.5]
    assert -0.5 <= v <= 0.5
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img
    
    H, S, V = img.convert('HSV').split()

    np_h = np.array(H, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(v * 255)
    H = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (H, S, V)).convert(input_mode)
    return img, label


def HorizontalFlip(img, label, v): # [0, 1]
    assert 0 <= v <= 1.0
    image = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return image, label


def Scale(img, label, v):  # [0.5, 2]
    assert 0.5 <= v <= 1.0
    W, H = img.size
    w, h = int(W * v), int(H * v)
    image = img.resize((w, h), PIL.Image.BILINEAR)
    label = label.resize((w, h), PIL.Image.NEAREST)
    

def Cutout(img, label, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img, label

    v = v * img.size[0]

    image, label = CutoutAbs(img, label, v)

    return image, label
    # x0 = np.random.uniform(w - v)
    # y0 = np.random.uniform(h - v)
    # xy = (x0, y0, x0 + v, y0 + v)
    # color = (127, 127, 127)
    # img = img.copy()
    # PIL.ImageDraw.Draw(img).rectangle(xy, color)
    # return img


def CutoutAbs(img, label, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img, label
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color_image = (255, 255, 255)
    color_label = (255)
    img = img.copy()
    label = label.copy()
    return PIL.ImageDraw.Draw(img).rectangle(xy, color_image), PIL.ImageDraw.Draw(label).rectangle(xy, color_label)


#def RandomCrop(img, label, W, H):
#    assert img.size == label.size
#    w, h = img.size
#    if (W, H) == (w, h): return img, label
#    if w < W or h < H:
#        scale = float(W) / w if w < h else float(H) / h
#        w, h = int(scale * w + 1), int(scale * h + 1)
#        img = img.resize((w, h), PIL.Image.BILINEAR)
#        label = label.resize((w, h), PIL.Image.NEAREST)
#    sw, sh = random.random() * (w - W), random.random() * (h - H)
#    crop = int(sw), int(sh), int(sw) + W, int(sh) + H
#    image = img.crop(crop)
#    label = label.crop(crop)
#    return image, label


#def RandomScale(img, label, v):
#    assert 0.75 < v < 2.0
#    W, H = img.size
#    w, h = int(W * v), int(H * v)
#    image = img.resize((w, h), PIL.Image.BILINEAR)
#    label = label.resize((w, h), PIL.Image.NEAREST)
#    return image, label


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=True):  # 16 operations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        #(Posterize, 4, 8),  # 9 # FastAutoaug
        (Posterize, 0, 8),
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
        (Hue, -0.5, 0.5)  # new 
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l

def randomaugment_list():  # 10 oeprations and their ranges for segmentation
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0, 1.0),

        # affine

        #(ShearX, 0., 0.3),  # 0
        #(ShearY, 0., 0.3),  # 1
        #(TranslateX, 0., 0.45),  # 2
        #(TranslateY, 0., 0.45),  # 3
        #(Rotate, 0, 30),  # 4

        # not colorjetter

        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 0, 8),  # 9

        # colorjetter

        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (AutoContrast, 0, 1),  # 5
        (Contrast, 0.1, 1.9),  # 10
        #(Hue, -0.5, 0.5), 

        # flip and scale

        #(Scale, 0.5, 1.0) # 15
        #(HorizontalFlip, 0, 1.0), # 14

        (Sharpness, 0.1, 1.9),  # 13

        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        image = im_lb['im']
        label = im_lb['lb']
        assert image.size == label.size
        W, H = self.size
        w, h = image.size

        if (W, H) == (w, h): return dict(im=image, lb=label)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            image = image.resize((w, h), PIL.Image.BILINEAR)
            label = label.resize((w, h), PIL.Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        image = image.crop(crop)
        label = label.crop(crop)
        return dict(im=image, lb=label)
                    

class RandomScale(object):
    def __init__(self, scales, *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        image = im_lb['im'] 
        label = im_lb['lb']      
        W, H = image.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        image = image.resize((w, h), Image.BILINEAR)
        label = label.resize((w, h), Image.NEAREST)
        return dict(im=image, lb=label)



class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = randomaugment_list()

    def __call__(self, im_lb):
        img = im_lb['im']
        label = im_lb['lb']
        ops = random.choices(self.augment_list, k=self.n)
        mag = random.randint(0, self.m)
        for op, minval, maxval in ops:
            val = (float(mag / self.m) * float(maxval - minval)) + minval
            img, label = op(img, label, val)

        return dict(im=img, lb=label)
