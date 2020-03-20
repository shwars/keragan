# Keragan Generator Script

import keragan
import argparse
import numpy as np
import cv2
import os
import glob
from keragan.utils import imread, imwrite

def imprint(sign,pos,img):
    overlay_image = sign[..., :3]
    mask = sign[..., 3:] / 255.0
    h,w = sign.shape[0],sign.shape[1]
    size_h,size_w = img.shape[0],img.shape[1]
    if pos == "lr":
        x,y=size_w-w,size_h-h
    elif pos=="ul":
        x,y = 0,0
    elif pos=="ur":
        x,y=size_w-w,0
    elif pos=="ll":
        x,y=0,size_h-h
    else:
        x,y = 0,0
    img[y:y+h, x:x+w] = (1.0 - mask) * img[y:y+h, x:x+w] + mask * overlay_image
    return img

if __name__ == '__main__':
    print("KeraGAN Signature Stamper, version {}".format(keragan.__version__))

    parser = argparse.ArgumentParser(description="KeraGAN Signature Stamper")

    parser.add_argument("--sign",help="Signature file",required=True)
    parser.add_argument("--size",help="Size of the signature",default=None,type=int)
    parser.add_argument("--relsize",help="Relative size as fraction of image size",type=float,default=None)
    parser.add_argument("--position",help="Position: UL,UR,LL,LR",default="LR")
    parser.add_argument("--input",help="Input directory",default=".")
    parser.add_argument("--output",help="Output directory",default=".")
    args = parser.parse_args()

    print(" + Loading signature...")
    sign = cv2.imread(args.sign,cv2.IMREAD_UNCHANGED)
    if args.size:
        sign = cv2.resize(sign,(args.size,args.size))

    print(" + Stamping...")

    for fn in glob.glob(os.path.join(args.input,"*")):
#    sz = args.signsize or int(args.size*0.1)
        print(fn)
        
        im = imread(fn)
        
        if im is None:
            continue
        if args.relsize:
            s = int(im.shape[0]*args.relsize)
            s = cv2.resize(sign,(s,s))
        else:
            s = sign
        res = imprint(s,args.position.lower(),im)
        imwrite(os.path.join(args.output,os.path.basename(fn)),res)

    print(" + Done!")
