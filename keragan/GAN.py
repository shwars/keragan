## Keras Implementation of GANs

import os
import keras
import cv2
import glob
import numpy as np 

class GAN():

    def __init__(self,width,height,model_path=None,samples_path=None,optimizer=None,lr=None,latent_dim=None):
        self.width = width
        self.height = height
        self.channels = 3
        self.img_shape = (self.height, self.width, self.channels)
        self.model_path = model_path
        self.lr = lr
        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.samples_path = samples_path
        self.init()

    @classmethod
    def from_args(cls,args):
        return cls(
            args.width,
            args.height,
            args.model_path,
            args.samples_path,
            args.optimizer,
            args.lr,
            args.latent_dim)

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
        if self.lr is None and self.optimizer is None:
            self.lr = 0.001
        self.optimizer = keras.optimizers.Adam(lr=self.lr, beta_1=0.5, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False) if self.optimizer is None else self.optimizer

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

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.discriminator.trainable = False
        
        z = keras.models.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        self.combined = keras.models.Model(z, valid) 
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    @staticmethod
    def create_from_generator_model(model):
        gan = GAN()
        gan.generator = model
        gan.discriminator = None
        gan.latent_dim = model.layers[0].input.shape[1].value
        return gan

    @staticmethod
    def create_from_generator_file(file):
        return GAN.create_from_generator_model(keras.models.load_model(file))

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
            confidence = self.discriminator.predict(gen_img) if self.discriminator is not None else 0
            # Rescale image to 0 - 255
            gen_img = ((0.5 * gen_img[0] + 0.5)*255).astype(np.uint8)
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
        size = self.width
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(8 * 8 * 2 * size, activation="relu", input_dim=self.latent_dim))
        model.add(keras.layers.Reshape((8, 8, 2 * size)))

        x = size
        while x>=16:
            model.add(keras.layers.UpSampling2D())
            model.add(keras.layers.Conv2D(x, kernel_size=(3,3), strides=1, padding="same"))
            model.add(keras.layers.BatchNormalization(momentum=0.8))
            model.add(keras.layers.Activation("relu"))
            x=x//2

        model.add(keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("tanh"))

        noise = keras.models.Input(shape=(self.latent_dim,))
        img = model(noise)

        return keras.models.Model(noise, img)


    def create_discriminator(self):
        
        model = keras.models.Sequential()

        i,x = 0,32
        while x<=2*self.width:
            if i==0:
                model.add(keras.layers.Conv2D(x, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
            else:
                model.add(keras.layers.Conv2D(x, kernel_size=3, strides=1, padding="same"))
            model.add(keras.layers.AveragePooling2D())
            model.add(keras.layers.BatchNormalization(momentum=0.8))
            model.add(keras.layers.LeakyReLU(alpha=0.2))
            model.add(keras.layers.Dropout(0.3))
            i+=1
            x*=2

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        img = keras.models.Input(shape=self.img_shape)
        validity = model(img)

        return keras.models.Model(img, validity)
    