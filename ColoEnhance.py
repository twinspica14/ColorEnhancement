#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import torchvision
import random
import os
import gc
from PIL import Image
import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import functools
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.exposure import rescale_intensity
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.utils.data as data
# import kornia
import types
import collections
import pandas as pd
from torch.utils import data
from random import shuffle
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import time
import timeit
import math
import pdb
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


torch.__version__


# In[3]:

class ExternalInputIterator(object):
    def __init__(self, batch_size=1, root_folder='./', height=512, shuffle_files=True):
        self.root = root_folder
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.datamode = 'data_512_5k' # train or test or self-define
        self.fine_height = height
        self.fine_width = height
        self.data_path = osp.join(self.root, self.datamode)
        self.files = os.listdir(self.data_path+'/train')  
        shuffle(self.files)
        self.range = np.arange(len(self.files))

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch_src = []
        batch_targ = []
        for _ in range(self.batch_size):
            file = self.files[self.i]
            f_src = open(osp.join(self.data_path,"train", file), 'rb')
            batch_src.append(np.frombuffer(f_src.read(), dtype = np.uint8))
            f_targ = open(osp.join(self.data_path,"target", file), 'rb')
            batch_targ.append(np.frombuffer(f_targ.read(), dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch_src, batch_targ)
          
    next = __next__
    
eii = ExternalInputIterator(batch_size=25, 
                            root_folder='./', 
                            height=512)
iterator = iter(eii)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.data_iterator = data_iterator
        self.src = ops.ExternalSource()
        self.targ = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB,)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)

        # resizing is *must* because loaded images maybe of different sizes
        # and to create GPU tensors we need image arrays to be of same size
#         self.res = ops.Resize(device="gpu", resize_x=512, resize_y=512, interp_type=types.INTERP_TRIANGULAR)
        self.res2 = ops.Resize(device="gpu", resize_x=256, resize_y=256, interp_type=types.INTERP_TRIANGULAR)
        self.flip_v = ops.Flip(device = "gpu", vertical = 1, horizontal = 0)
        self.flip_h = ops.Flip(device = "gpu", vertical = 0, horizontal = 1)

    def define_graph(self):
        self.jpegs_src = self.src()
        self.jpegs_targ = self.targ()
        images_src = self.decode(self.jpegs_src)
        images_real = self.decode(self.jpegs_src)
        images_targ = self.decode(self.jpegs_targ)

        output_src = images_src
        output_real = self.res2(images_real)
        output_targ = images_targ
        
        if random.random() > 0.5:
            output_src = self.flip_h(output_src)
            output_real = self.flip_h(output_real)
            output_targ = self.flip_h(output_targ)
        else:
            output_src = self.flip_h(output_src)
            output_real = self.flip_h(output_real)
            output_targ = self.flip_h(output_targ)
        return (output_src,output_real, output_targ)

    def iter_setup(self):
        # the external data iterator is consumed here and fed as input to Pipeline
        src, targ = self.data_iterator.next()
        self.feed_input(self.jpegs_src, src)
        self.feed_input(self.jpegs_targ, targ)


# In[4]:


pipe = ExternalSourcePipeline(data_iterator=iterator, batch_size=25, num_threads=4, device_id=0)
pipe.build()
dali_iter = DALIGenericIterator([pipe], ['src','real', 'targ'],dynamic_shape=True,size=5000, auto_reset=True)





# In[5]:


# Code taken from kornia
def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    See :class:`~kornia.color.RgbToXyz` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ.

    Returns:
        torch.Tensor: XYZ version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack((x, y, z), -3)

    return out

def xyz_to_lab(image: torch.Tensor) -> torch.Tensor:
    
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    l: torch.Tensor = (116 * x) - 16
    a: torch.Tensor = 500 * (x - y)
    b: torch.Tensor = 200 * (y - z)

    out: torch.Tensor = torch.stack((l, a, b), -3)
    
    return out

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v

    s[torch.isnan(s)] = 0.

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h
    return torch.stack([h, s, v], dim=-3)   

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to LAB.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    

    return xyz_to_lab(rgb_to_xyz(image))

def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to grayscale.

    See :class:`~kornia.color.RgbToGrayscale` for details.

    Args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
    
# In[6]:


class Transformer(nn.Module):
    def __init__(self,Norm=nn.InstanceNorm2d, Act=nn.ReLU(True),in_features=64, in_channels=3, n_batch=5):
        super(Transformer, self).__init__()
#         Act = functools.partial(Act)
        Middle_layer = [nn.Conv2d(in_features*2, in_features*2, 3,2),
                        Norm(in_features*2),
                        Act,
                        nn.Dropout2d(0.5)]
        ml = []
        for i in range(0,3):
            ml+=Middle_layer
            
        self.Middle_layer = nn.Sequential(*ml)

        self.Input_layer = nn.Sequential(nn.Conv2d(in_channels, in_features, 5,2),
                Act,
                nn.Conv2d(in_features, in_features*2, 3),
                Norm(in_features*2),
                Act
                 )
        
        self.Average_layer = nn.Sequential(
            nn.Conv2d(in_features*2, in_features*2, 3,2),
            Norm(in_features*2),
            Act,
            nn.AdaptiveAvgPool2d((6, 6)),
            Act,
            nn.Dropout2d(0.5)
        )
        
        self.Finale_layer = nn.Sequential(
            nn.Conv2d(in_features*2, in_channels, 1),
            Act,
            nn.Dropout2d(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(36,16),
            Act,
            nn.Linear(16,10)
        )
        self.l = nn.Linear(5*n_batch,1*n_batch)
    def forward(self, x,y):
        b1_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b1 = self.classifier(b1_.view(b1_.size(0),3,-1))
        b2_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b2 = self.classifier(b2_.view(b2_.size(0),3,-1))
        b3_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b3 = self.classifier(b3_.view(b3_.size(0),3,-1))
        b4_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b4 = self.classifier(b4_.view(b4_.size(0),3,-1))
        b5_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b5 = self.classifier(b5_.view(b5_.size(0),3,-1))
        
        concat = torch.cat([b1,b2,b3,b4,b5],0)
#         print(concat.shape)
        h_theta = self.l(concat.T)
        h_theta = h_theta.permute(0,2,1)
        V_p = self.get_param(y)
        output= torch.einsum("abcde,abf->bfde", (V_p, h_theta))
        return output+y

    
    def get_param(self,x):
        R = x[:,:1,:,:]
        G = x[:,1:2,:,:]
        B = x[:,2:,:,:]
        C = torch.ones_like(R)
        return torch.stack([R, G, B, torch.pow(R,2), torch.pow(G,2), torch.pow(B,2), R * G, G * B, B * R, C])        


# In[7]:


class Loss():
    def __init__(self, col_hsv=False, col_lab=True, col_gray=True):
        self.transform = list()
        if col_hsv:
            self.transform.append(rgb_to_hsv)
        if col_lab:
            print("lab")
            self.transform.append(rgb_to_lab)
        if col_gray:
            print("gray")
            self.transform.append(rgb_to_grayscale)

        self.criterion = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
#         self.psnr = kornia.losses.PSNRLoss(2)
#         self.ssim = kornia.losses.SSIM(5, reduction='none')
    def __call__(self,img_input,img_target):
        loss = 0.0
        input = self.transform[0](img_input)
        target = self.transform[0](img_target)
        loss+=20*(self.criterion(input[:,0,:,:],target[:,0,:,:])+self.criterion(input[:,1,:,:],target[:,1,:,:])+self.criterion(input[:,2,:,:],target[:,2,:,:]))
        
#         loss+=self.criterion(self.transform[1](img_input),self.transform[1](img_target))
        
        loss+=self.huber(self.transform[1](img_input),self.transform[1](img_target))
        return loss#+self.psnr(img_input,img_target)+self.ssim(img_input,img_target)


# In[8]:


epoch = 0
n_epochs = 500
decay_epoch = 30
batchSize = 25
lr = 0.0009


# In[9]:


criterion = Loss()


# In[10]:


model = Transformer(n_batch=batchSize).cuda()


# In[11]:


lambda1 = lambda epoch: 0.55 ** (epoch)
optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.5, 0.999))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


# In[12]:


def trans(img,real,targ):
    img = img.permute(0,3,1,2)
    real = real.permute(0,3,1,2)
    targ = targ.permute(0,3,1,2)
    mean_512 = torch.ones_like(img)*0.5*255
    std_512 = torch.ones_like(img)*0.5*255
    mean_256 = torch.ones_like(real)*0.5*255
    std_256 = torch.ones_like(real)*0.5*255
    
    img = (img-mean_512)/std_512
    real = (real-mean_256)/std_256
    targ = (targ-mean_512)/std_512
    
    return img,real,targ


# In[22]:


model.train()
for epoch in range(0, n_epochs):
    gc.collect()
#     test = iter(test_loader)
    avg_loss = 0
    for i, it in enumerate(dali_iter):
        start = time.time()
        batch_data = it[0]
        img,real,targ = batch_data["src"],batch_data["real"], batch_data["targ"]
        
        img,real,targ = trans(img,real,targ) 
        
        optimizer.zero_grad()
        
        #Inverse identity
        res = model(real,img) # previously for res 3, mask,src
        loss = criterion(res,targ)        
        loss.sum().backward()
        
        optimizer.step() 
        

        #############################################
        avg_loss = (avg_loss+loss.sum().item())/(i+1) 
          
        if (i + 1) % 200 == 0:
            stop = time.time()
            print('Time: ', stop - start)
            with open('cpe_64.txt', 'a') as f:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) LR: (%0.0007f) Time: (%0.0007f)" % (epoch, i + 1, 200, avg_loss, optimizer.param_groups[0]['lr'],stop - start),file=f)
            
        
        if (i + 1) % 200 == 0:            
            pic = (torch.cat([img,res, targ], dim=0).data + 1) / 2.0
            save_dir = "./results"
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, 200), nrow=3)
    if (epoch+1)%1==0:
        torch.save(model.state_dict(), './models/model_64_{}.pth'.format(epoch))
    # Update learning rates
    if (epoch+1)%30==0 and epoch<320:
        lr_scheduler.step()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




