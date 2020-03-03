import numpy as np

class GANTrainer():

    def __init__(self, image_dataset, gan, args=None, no_samples = None, batch_size=128, print_interval=10, save_interval=50, save_img_interval=50, epochs=100):
        self.data = image_dataset
        self.gan = gan
        if args:
            self.batch_size = args.batch_size
            self.save_interval = args.save_interval
            self.save_img_interval = args.save_img_interval
            self.epochs = args.epochs
            self.no_samples = args.no_samples
            self.print_interval = args.print_interval
        else:
            self.batch_size = batch_size
            self.save_interval = save_interval
            self.save_img_interval = save_img_interval
            self.print_interval = print_interval
            self.epochs = epochs
            self.no_samples = no_samples

        if self.gan.height!=self.data.height or self.gan.width!=self.data.width:
            raise "Image size of GAN and ImageDataset should be the same"

    def train(self,callback=None):
        # ones = label for real images
        # zeros = label for fake images
        ones = np.ones((self.batch_size, 1)) 
        zeros = np.zeros((self.batch_size, 1))

        # create some noise to track AI's progression
        if self.no_samples:
            self.noise_pred = np.random.normal(0, 1, (self.no_samples, 1, self.gan.latent_dim))

        for ep in range(self.epochs):

            # Select a random batch of images in dataset
            imgs = self.data.get_batch(self.batch_size)
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.batch_size, self.gan.latent_dim))
            gen_imgs = self.gan.generator.predict(noise)   

            # Train the discriminator with generated images and real images
            d_loss_r = self.gan.discriminator.train_on_batch(imgs, ones)
            d_loss_f = self.gan.discriminator.train_on_batch(gen_imgs, zeros)
            d_loss = np.add(d_loss_r , d_loss_f)*0.5

            # Trains the generator to fool the discriminator
            g_loss = self.gan.combined.train_on_batch(noise, ones)

            #print loss and accuracy of both trains
            if ep % self.print_interval == 0:
                print ("%d D loss: %f, acc.: %.2f%% G loss: %f" % (self.gan.epoch, d_loss[0], 100*d_loss[1], g_loss))

            if ep % self.save_img_interval == 0 and self.no_samples:
                print("Saving images...", end='')
                self.gan.write_sample_images(self.noise_pred)
                print("done")
            
            if ep % self.save_interval == 0:
                self.gan.save()
            
            if callback:
                callback(self)

            self.gan.epoch+=1
