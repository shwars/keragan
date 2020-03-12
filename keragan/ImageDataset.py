# Keras implementation of GANs
# Image Dateset Class

import os
import cv2
import glob
import tqdm
import numpy as np
from .utils import show_images, crop_resize

class ImageDataset():

    def __init__(self,path,height,width,aspect_variance=None,save_npy_path=None,ignore_smaller=True,limit=None,crop=False):
        self.path = path
        self.height = height
        self.width = width
        self.aspect_variance = aspect_variance
        self.save_npy_path = save_npy_path
        self.ignore_smaller = ignore_smaller
        self.limit = limit
        self.crop = crop

    def __init__(self,args):
        self.path = args.path
        self.height = args.height
        self.width = args.width
        self.aspect_variance = args.aspect_variance
        self.save_npy_path = args.save_npy_path
        self.ignore_smaller = args.ignore_smaller
        self.limit = args.limit
        self.crop = args.crop

    def load(self):

        if self.save_npy_path is not None and os.path.isfile(self.save_npy_path):
            print('Loading pre-processed dataset...',end='')
            X_train = np.load(self.save_npy_path)
            print("done")
            return X_train

        print("Loading original dataset...",end='')
        
        asp = self.height/self.width
        asp_mi = 0
        asp_ma = 1000
        if self.aspect_variance:
            asp_mi = asp*(1-self.aspect_variance)
            asp_ma = asp*(1+self.aspect_variance)

        X_train = []
        dos = glob.glob(os.path.join(self.path,'*'))

        n = 0
        for i in tqdm.tqdm(dos):
            try:
                img = cv2.imread(i)
                asp = img.shape[0]/img.shape[1]
                if self.ignore_smaller and (img.shape[0]<self.height or img.shape[1]<self.width):
                    print("Image {} too small - skipping")
                elif self.aspect_variance is not None and not(asp_mi<img.shape[0]/img.shape[1]<asp_ma):
                    print("Image {} too skew - skipping")
                else:
                    if self.crop:
                        img = crop_resize(img,self.width, self.height)
                    else:
                        img = cv2.resize(img,(self.width, self.height))
                    X_train.append(img)
                    n+=1
                    if self.limit is not None and n>self.limit:
                        break
            except:
                print("Error loading {} - skipping".format(i))
                

        X_train = np.array(X_train)

        # Rescale dataset to -1 - 1
        X_train = X_train / 127.5 - 1
        if self.save_npy_path:
            print('...saving...',end='')
            np.save(self.save_npy_path,X_train)
        print('done, loaded {} images'.format(n))
        
        self.data = X_train

        return X_train

    def sample_images(self,n=10):
        imgs = (self.get_batch(n).copy()+1)/2.0
        show_images(imgs)

    def get_batch(self,batch_size):
            idx = np.random.randint(0, self.data.shape[0], batch_size)
            imgs = self.data[idx]
            return imgs



