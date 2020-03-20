# Keragan Generator Script

import argparse
import keragan
import numpy as np
import cv2
import os

if __name__ == '__main__':
    print("KeraGAN Generator, version {}".format(keragan.__version__))

    parser = argparse.ArgumentParser(description="KeraGAN Generator")

    parser.add_argument("--file",help="Generator model to load",required=True)
    parser.add_argument("--n",help="Number of images to generate",type=int,default=1)
    parser.add_argument("--out",help="Output directory",default=".")
    parser.add_argument("--prefix",help="File prefix to use",default="")
    parser.add_argument("--ext",help="Image type (jpg or png)",default="jpg")    
    parser.add_argument("--colour",help="rgb, bgr or both",default="both")
    parser.add_argument("--batch_size",help="Minibatch size",default=32)
    args = parser.parse_args()

    print(" + Loading model...")
    gan = keragan.GAN.create_from_generator_file(args.file)

    print(" + Generating...")
    i = 0
    while i<args.n:
        left = args.n - i
        x = min(left,args.batch_size)
        res = gan.sample_images(n=x)
        for img,conf in res:
            if args.colour in ["bgr","both"]:
                cv2.imwrite(os.path.join(args.out,"{}{}_bgr.{}".format(args.prefix,i,args.ext)),img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if args.colour in ["rgb","both"]:
                cv2.imwrite(os.path.join(args.out,"{}{}_rgb.{}".format(args.prefix,i,args.ext)),img)
            i+=1
    print(" + Done!")
