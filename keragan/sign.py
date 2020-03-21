# Keragan Generator Script

import keragan
import argparse
import numpy as np
import cv2
import os
import random
import glob
import imutils
from keragan.utils import imread, imwrite

from sklearn.cluster import KMeans
from collections import Counter


def percent(value,percents):
    return int(float(value)*float(percents)/100.0)


def get_dominant_color(image, k=4, image_processing_size = None):
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_color

def get_av_color(img):
    m = img[:,:,3]==0
    ms = np.zeros_like(img[:,:,0:3],dtype=np.bool)
    ms[:,:,0] = m
    ms[:,:,1] = m
    ms[:,:,2] = m
    ma = np.ma.MaskedArray(img[:,:,0:3].astype(np.float32),ms)
    return ma.mean(axis=(0,1))

def get_sign(signs,im,args):
    if len(signs)==1:
        return signs[0]
    elif args.stamp_selection == "color":
        dc = get_dominant_color(im)
        l = sorted(signs,key=lambda x: -np.sum(np.square(dc-get_av_color(x))))
        return random.choice(l[0:args.rand_stamp])
    else:
        return random.choice(signs)

def get_pos(size_w, size_h, w, h, rcx, rcy, pos):
    if pos == "lr":
        x,y=size_w-w-rcx,size_h-h-rcy
    elif pos=="ul":
        x,y = rcx,rcy
    elif pos=="ur":
        x,y=size_w-w-rcx,rcy
    elif pos=="ll":
        x,y=rcx,size_h-h-rcy
    else:
        x,y = 0,0
    return x,y

def imprint(signs,args,img):
    ra = random.randint(-args.rand_rot,args.rand_rot)
    size = args.size
    rand_size = args.rand_size
    rand_coord = args.rand_coord
        
    size_h,size_w = img.shape[0],img.shape[1]

    if args.unit == "percent":
        size = percent(size_w,size) 
        rand_size = percent(size,rand_size)
        rand_coord = percent(size_w,rand_coord)

    rcx = random.randint(0,rand_coord)
    rcy = random.randint(0,rand_coord)
    size += random.randint(-rand_size,rand_size)

    pos = args.position.lower()
    x,y = get_pos(size_w,size_h,size,size,rcx,rcy,pos)

    sign = get_sign(signs,img[y:y+size,x:x+size],args)

    s = cv2.resize(sign,(size,size))
    s = s if ra==0 else imutils.rotate_bound(s,ra)
    overlay_image = s[..., :3]
    mask = s[..., 3:] / 255.0
    h,w = s.shape[0],s.shape[1]
    x,y = get_pos(size_w,size_h,w,h,rcx,rcy,pos)
    img[y:y+h, x:x+w] = (1.0 - mask) * img[y:y+h, x:x+w] + mask * overlay_image
    return img

if __name__ == '__main__':
    print("KeraGAN Signature Stamper, version {}".format(keragan.__version__))

    parser = argparse.ArgumentParser(description="KeraGAN Signature Stamper")

    parser.add_argument("--sign",help="Signature file or directory",required=True)
    parser.add_argument("--size",help="Size of the signature",default=None,type=int)
    parser.add_argument("--position",help="Position: UL,UR,LL,LR",default="LR")
    parser.add_argument("--input",help="Input directory",default=".")
    parser.add_argument("--output",help="Output directory",default=".")
    parser.add_argument("--rand_coord",help="Coordinate randomization",type=int,default=0)
    parser.add_argument("--rand_rot",help="Rotation randomization (deg)",type=int,default=0)
    parser.add_argument("--rand_size",help="Size randomization",type=int,default=0)
    parser.add_argument("--rand_stamp",help="No of stamp randomization",type=int,default=1)
    parser.add_argument("--stamp_selection",help="Algorithm of stamp selection [random, color]",default="color",choices=["random","color"])
    parser.add_argument("--unit",help="Size measurement units: pix or percent",default="pix",choices=["pix","percent"])

    args = parser.parse_args()

    print(" + Loading signature(s)...")
    if os.path.isdir(args.sign):
        sign = [cv2.imread(f,cv2.IMREAD_UNCHANGED) for f in glob.glob(os.path.join(args.sign,'*'))]
        sign = [x for x in sign if x is not None]
    else:
        sign = [cv2.imread(args.sign,cv2.IMREAD_UNCHANGED)]

    print(" + Stamping...")
    for fn in glob.glob(os.path.join(args.input,"*")):
#    sz = args.signsize or int(args.size*0.1)
        im = imread(fn)        
        if im is None:
            continue
        res = imprint(sign,args,im)
        imwrite(os.path.join(args.output,os.path.basename(fn)),res)

    print(" + Done!")
