# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import numpy as np
import os
import torch.utils.data as data
import imageio
from .styleaug import StyleAugmentor
import imgaug.augmenters as iaa
from graphs.models import net
from utils.utils import *

from datasets.cityscapes_Dataset import City_Dataset
from pathlib import Path
import random
import scipy.misc
from datasets.augmentations import RandAugment

from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEBUG = False

class GTA5_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/GTA5',
                 list_path='./datasets/GTA5',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 transform=None, 
                 limits=None,
                 adain=None,
                 styleaug=None,
                 fda=None,
                 imgaug=None):

        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        if DEBUG: print('DEBUG: GTA {0:} dataset path is {1:}'.format(self.split, self.list_path))
        self.base_size=base_size
        self.crop_size=crop_size
        if DEBUG: print('DEBUG: GTA {0:} dataset image size is {1:}'.format(self.split, self.crop_size))

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        self.limits = args.limits
        self.imgaug = args.imgaug
        self.adain = args.adain
        self.styleaug = args.styleaug
        self.fda = args.fda
        self.autoaug=args.autoaug
         
        ###
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        #self.image_filepath = os.path.join(self.data_path, "images")
        #self.gt_filepath = os.path.join(self.data_path, "labels")
        self.items = [id.strip() for id in open(item_list_filepath)]
        ###

        ignore_label = -1

        ###
        # self.id_to_trainid = {i: i for i in range(-1, 19)}
        # self.id_to_trainid[255] = ignore_label

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        ###

        self.class_16 = False
        self.class_13 = False

        if DEBUG: print('DEBUG: GTA {0:} -> item_list_filepath: {1:} , first item: {2:}'.format(self.split, item_list_filepath, self.items[0]))
        if DEBUG: print("{} num images in GTA5 {} set have been loaded.".format(len(self.items), self.split))

        #self.fda = fda
        if self.fda:
            self.fda = args.fda[0]
            self.fda_L = args.fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        if self.styleaug:
            self.augmentor = StyleAugmentor()

        if self.autoaug:
            self.autoaugmentor = RandAugment(3,14)

    def __getitem__(self, item):
        ###
        #id_img, id_gt = self.items[item].strip('\n').split(' ')
        #image_path = self.data_path + id_img
        id_img = self.items[item]#.strip('\n').split(' ')
        image_path = self.data_path + "/images/" + id_img
        image = Image.open(image_path).convert("RGB")
        if item == 0 and DEBUG: print('DEBUG: GTA {0:} -> image_path: {1:}'.format(self.split, image_path))
        ###

        ###
        #gt_image_path = self.data_path + id_gt.split(".")[0] + "_label.png"
        gt_image_path = self.data_path + "/labels/" + id_img.split('.')[0] + "_label.png"

        # gt_image = imageio.imread(gt_image_path, format='PNG-FI')[:, :, 0]
        # gt_image = Image.fromarray(np.uint8(gt_image))
        #
        gt_image = Image.open(gt_image_path)

        tmp_gt = np.asarray(gt_image, np.float32)
        u = np.unique(tmp_gt)
        if len(u) > 0:
            image = image.resize(self.crop_size, Image.BICUBIC)
            gt_image = gt_image.resize(self.crop_size, Image.NEAREST)

            if item == 0 and DEBUG: print('DEBUG: GTA {0:} -> image_path: {1:}'.format(self.split, image_path))
            ###

            """
            if self.adain:
                style_choice = random.randint(0, len(self.style_paths) - 1)
                content = self.content_tf(image)
                style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
                style = style.to("cuda:0" if torch.cuda.is_available() else "cpu").unsqueeze(0)
                content = content.to("cuda:0" if torch.cuda.is_available() else "cpu").unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
                output = output.cpu().squeeze(0).numpy()
                img_aug = toimage(output)
                #image.save('temp/adain.png')
            """

            if self.fda:
                #print("FDA processing...")
                source = np.asarray(image, np.float32)
                if self.fda == 'random':
                    target = np.random.uniform(1, 255, source.shape)
                else:
                    choice = random.randint(0, len(self.tgt_paths) - 1)
                    target = Image.open(self.tgt_paths[choice])
                    target = target.resize(self.base_size, Image.BICUBIC)

                target = np.asarray(target, np.float32)
                source = source.transpose((2,0,1))
                target = target.transpose((2,0,1))

                output = FDA_source_to_target_np(source, target, L=self.fda_L)
                img_aug = toimage(output, channel_axis=0, cmin=0, cmax=255)
                gt_image_aug = gt_image
                img_aug.save('temp.png') 

            """
            if self.styleaug:
                imtorch = transforms.ToTensor()(image).unsqueeze(0)
                imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    imrestyled  = self.augmentor(imtorch)
                imrestyled = imrestyled.cpu().squeeze(0).numpy()
                img_aug = toimage(imrestyled)
                #image.save('temp/test.png')
            """

            """
            if self.autoaug:
                im_lb = dict(im=image, lb=gt_image)
                im_lb = self.autoaugmentor(im_lb)
                img_aug = im_lb['im']
                gt_image_aug = im_lb['lb']
                #img_aug.save('temp/img_aug.png')
            """

            if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
                image, gt_image = self._train_sync_transform(image, gt_image)
                img_aug, gt_img_aug = self._train_sync_transform(img_aug, gt_image_aug)

            else:
                image, gt_image = self._val_sync_transform(image, gt_image)
                img_aug, gt_img_aug = image, gt_image


            return image, gt_image, img_aug, gt_img_aug, id_img

class GTA5_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = GTA5_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training)

        if self.args.split == "train" or self.args.split == "trainval" or self.args.split =="all":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        elif self.args.split =="val" or self.args.split == "test":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            raise Warning("split must be train/val/trainavl/test/all")

        val_split = 'val' if self.args.split == "train" else 'test'
        val_set = GTA5_Dataset(args, 
                            data_root_path=args.data_root_path,
                            list_path=args.list_path,
                            split=val_split,
                            base_size=args.base_size,
                            crop_size=args.crop_size,
                            training=False)
        self.val_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size


        
