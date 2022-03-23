import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample

from torch.nn import functional as F

'''
def load_img(image_path, scale):
    HR = []
    HR = []
    for img_num in range(7):
        GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR
'''

def load_img(image_path, scale):
    HR = []
    HR = []

    '''
    if random.random()<0.5:
        for img_num in range(7):
            GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(7-img_num))).convert('RGB'), scale)
            #box = (100,100,356,356)#228
            #GT_temp = GT_temp.crop(box) 
            #print(GT_temp.format, GT_temp.size, GT_temp.mode)
            HR.append(GT_temp)
        return HR
    else:
        for img_num in range(7):
            GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            #box = (100,100,356,356)#228
            #GT_temp = GT_temp.crop(box) 
            #print(GT_temp.format, GT_temp.size, GT_temp.mode)
            HR.append(GT_temp)
        return HR
    '''

    tt=random.random()
    tt=int(tt*192+0.5)
    box = (tt,0,tt+256,256)#228    
    for img_num in range(7):
        GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        GT_temp = GT_temp.crop(box) 
        #print(GT_temp.format, GT_temp.size, GT_temp.mode)
        HR.append(GT_temp)
    return HR

def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def train_process(GH, flip_h=True, rot=True, converse=True): 
    if random.random() < 0.5 and flip_h: 
        GH = [ImageOps.flip(LR) for LR in GH]
    if rot:
        if random.random() < 0.5:
            GH = [ImageOps.mirror(LR) for LR in GH]
    return GH

class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] 
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] 
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale) 
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        L=len(GT)
        #GT, L = load_img(self.image_filenames[index], self.scale, self.L, image_pad=True) 
        GT = [np.asarray(HR) for HR in GT] 
        GT = np.asarray(GT)
        #if self.scale == 4:
        #    GT = np.lib.pad(GT, pad_width=((0,0), (2*4,2*4), (2*4,2*4), (0,0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = LR.permute(1,0,2,3)
        GT = GT.permute(1,0,2,3) # [T,C,H,W]
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, L, GT_D, GT_S
        
    def __len__(self):
        return len(self.image_filenames) 

