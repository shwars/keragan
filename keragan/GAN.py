## Keras Implementation of GANs

import os
import keras
import cv2
import glob
import numpy as np 

class GAN():

    def __init__(self,width,height,model_path=None,samples_path=None,optimizer=None,latent_dim=None):
        self.width = width
        self.height = height
        self.channels = 3
        self.img_shape = (self.height, self.width, self.channels)
        self.model_path = model_path
        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.samples_path = samples_path
        self.init()

    def __init__(self,args):
        self.width = args.width
        self.height = args.height
        self.channels = 3
        self.img_shape = (self.height, self.width, self.channels)
        self.model_path = args.model_path
        self.optimizer = args.optimizer
        self.latent_dim = args.latent_dim
        self.samples_path = args.samples_path
        self.init()

    def __figure_epoch(self):
        mx = -1
        for fn in glob.glob(os.path.join(self.model_path,'dis_*.h5')):
            n = os.path.basename(fn)
            n = int(n[4:-3])
            if n>mx:
                mx = n
        if mx==-1:
            return None
        else:
            return mx

    def init(self):
        self.optimizer = keras.optimizers.Adam(0.005) if self.optimizer is None else self.optimizer

        if self.model_path:
            os.makedirs(self.model_path,exist_ok=True)
        if self.samples_path:
            os.makedirs(self.samples_path,exist_ok=True)

        self.epoch = self.__figure_epoch()

        if self.epoch is not None:
            print("Loading existing models...")
            self.load(self.epoch)
            self.epoch+=1
        else:
            # build discriminator, generator
            self.discriminator = self.create_discriminator()
            self.generator = self.create_generator()
            self.epoch = 0

        self.discriminator.trainable = False
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        z = keras.models.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        self.combined = keras.models.Model(z, valid) 
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def create_generator(self):
        raise "Abstract GAN Class cannot be instantiated"

    def create_discriminator(self):
        raise "Abstract GAN Class cannot be instantiated"

    def save(self):
        self.discriminator.save(os.path.join(self.model_path,'dis_'+str(self.epoch)+'.h5'))
        self.generator.save(os.path.join(self.model_path,'gen_'+str(self.epoch)+'.h5'))

    def load(self,epoch=None):
        if epoch is None:
            epoch = self.epoch
        self.discriminator = keras.models.load_model(os.path.join(self.model_path,'dis_'+str(epoch)+'.h5'))
        self.generator = keras.models.load_model(os.path.join(self.model_path,'gen_'+str(epoch)+'.h5'))
        self.epoch = epoch

    def sample_images(self,noise_vector=None,n=10):
        if noise_vector is None:
            noise_vector = np.random.normal(0, 1, (n, 1, self.latent_dim))
        res = []
        for i,s in enumerate(noise_vector):
            gen_img = self.generator.predict(s)
            confidence = self.discriminator.predict(gen_img)
            # Rescale image to 0 - 255
            gen_img = (0.5 * gen_img[0] + 0.5)*255
            res.append((gen_img,confidence))
        return res

    def write_sample_images(self,noise_vector=None,n=10):
        res = self.sample_images(noise_vector=noise_vector,n=n)
        for i,v in enumerate(res):
            gen_img,confidence = v
            cv2.imwrite(os.path.join(self.samples_path,'img_%d_%d_%f.png'%(i,self.epoch, confidence)), gen_img)
    
class DCGAN(GAN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def create_generator(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(8 * 8 * 512, activation="relu", input_dim=self.latent_dim))
        model.add(keras.layers.Reshape((8, 8, 512)))

        for x in [512,256,128,64,32,16,8]:
            model.add(keras.layers.Conv2D(x, kernel_size=(3,3), padding="same"))
            model.add(keras.layers.Activation("relu"))
            model.add(keras.layers.BatchNormalization(momentum=0.8))
            if x!=8:
                model.add(keras.layers.UpSampling2D())

        model.add(keras.layers.Conv2D(self.channels, kernel_size=(1,1), padding="same"))
        model.add(keras.layers.Activation("tanh"))

        noise = keras.models.Input(shape=(self.latent_dim,))
        img = model(noise)

        return keras.models.Model(noise, img)


    def create_discriminator(self):
        
        model = keras.models.Sequential()

        for i,x in enumerate([32,64,128,256]):
            if i==0:
                model.add(keras.layers.Conv2D(x, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
            else:
                model.add(keras.layers.Conv2D(x, kernel_size=3, strides=2, padding="same"))

            model.add(keras.layers.BatchNormalization(momentum=0.8))
            model.add(keras.layers.LeakyReLU(alpha=0.2))
            model.add(keras.layers.AveragePooling2D())
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        img = keras.models.Input(shape=self.img_shape)
        validity = model(img)

        return keras.models.Model(img, validity)

    