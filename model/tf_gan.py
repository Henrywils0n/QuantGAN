from preprocess.acf import *
import numpy as np
import tensorflow as tf

from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.utils import Progbar
from keras.models import load_model, Model
from keras.layers import Input, Concatenate
from tensorflow import convert_to_tensor
from math import floor, ceil
from scipy.stats import wasserstein_distance, entropy

#Alpha GAN generator and discriminator loss functions from Justin Veiner, github.com/justin-veiner/MASc
#from alpha_loss import dis_loss_alpha, gen_loss_alpha
def dis_loss_alpha(real_predicted_labels, fake_predicted_labels, alpha_d):
    # FOR ALPHA_D <= 1, ALPHA_G MUST BE IN RANGE (ALPHA_D/(ALPHA_D+1), infinity)
    # FOR ALPHA_D > 1, ALPHA_G MUST BE IN RANGE (ALPHA_D/2, ALPHA_D)

    print("Alpha_D Value: " + str(alpha_d))

    """
    fake_predicted_labels: fake predicted values
    real_predicted_labels: real predicted values
    alpha_d: alpha parameter for the discriminator loss function (positive float, default 3.0)
    gp: add a gradient penalty to the discriminator's loss function (bool, default False)
    gp_coef: coefficient for gradient penalty term if added (float, default 5.0)
    img: I DO NOT KNOW WHAT THIS IS. Commented out this section. Think has something to do with the real training data.
    """
    sigmoid_output_real = tf.nn.sigmoid(real_predicted_labels)
    sigmoid_output_fake = tf.nn.sigmoid(fake_predicted_labels)

    real_expr = tf.math.pow(sigmoid_output_real, ((alpha_d-1)/alpha_d)*tf.ones_like(sigmoid_output_real))
    real_loss = tf.math.reduce_mean(real_expr)
    fake_expr = tf.math.pow(1 - sigmoid_output_fake, ((alpha_d-1)/alpha_d)*tf.ones_like(sigmoid_output_fake))
    fake_loss = tf.math.reduce_mean(fake_expr)
 
    loss_expr = -(alpha_d/(alpha_d - 1))*(real_loss + fake_loss - 2.0)

    return loss_expr

def gen_loss_alpha(fake_predicted_labels, alpha_g):
    """
    fake_predicted_labels: fake predicted values
    alpha_g: alpha parameter for the generator loss function (positive float, default 3.0)
    l1: I DO NOT KNOW WHAT THIS IS. Set to False by default. Liekly somethign to do with L1
    """
    # FOR ALPHA_D <= 1, ALPHA_G MUST BE IN RANGE (ALPHA_D/(ALPHA_D+1), infinity)
    # FOR ALPHA_D > 1, ALPHA_G MUST BE IN RANGE (ALPHA_D/2, ALPHA_D)
    print("Alpha_G Value: " + str(alpha_g))
    
    sigmoid_output = tf.nn.sigmoid(fake_predicted_labels)
    
    fake_expr = tf.math.pow(1 - sigmoid_output, ((alpha_g-1)/alpha_g)*tf.ones_like(sigmoid_output))
    fake_loss = tf.math.reduce_mean(fake_expr)
    loss_expr = (alpha_g/(alpha_g - 1))*(fake_loss - 2.0)

    return loss_expr

class GAN:
    """ Generative adverserial network class.

    Training code for a standard DCGAN using the Adam optimizer.
    Code taken in part from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
    """    
    
    """
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)
    """
    
    def discriminator_loss(self, real_output, fake_output):
        if self.alpha_d == 1:
            real_loss = self.loss(tf.ones_like(real_output), real_output)
            fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss
        return dis_loss_alpha(real_output, fake_output, self.alpha_d)
    
    def generator_loss(self, fake_output):
        if self.alpha_g == 1:
            return self.loss(tf.ones_like(fake_output), fake_output)
        return gen_loss_alpha(fake_output, self.alpha_g)

    def __init__(self, discriminator, generator, training_input, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, from_logits=True):
        """Create a GAN instance

        Args:
            discriminator (tensorflow.keras.models.Model): Discriminator model.
            generator (tensorflow.keras.models.Model): Generator model.
            training_input (int): input size of temporal axis of noise samples.
            lr_d (float, optional): Learning rate of discriminator. Defaults to 1e-4.
            lr_g (float, optional): Learning rate of generator. Defaults to 3e-4.
            epsilon (float, optional): Epsilon paramater of Adam. Defaults to 1e-8.
            beta_1 (float, optional): Beta1 parameter of Adam. Defaults to 0.
            beta_2 (float, optional): Beta2 parameter of Adam. Defaults to 0.9.
            from_logits (bool, optional): Output range of discriminator, logits imply output on the entire reals. Defaults to True.
        """
        self.alpha_d = 1
        self.alpha_g = 1
        self.discriminator = discriminator
        self.generator = generator
        self.noise_shape = [self.generator.input_shape[1], training_input, self.generator.input_shape[-1]]

        self.loss = BinaryCrossentropy(from_logits=from_logits)

        self.generator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def train(self, data, batch_size, n_batches, real_dist, return_divergence=False):
        """training function of a GAN instance.
        Args:
            data (4d array): Training data in the following shape: (samples, timesteps, 1).
            batch_size (int): Batch size used during training.
            n_batches (int): Number of update steps taken.
        """ 
        progress = Progbar(n_batches)
        train_divergence = []

        for n_batch in range(n_batches):
            # sample uniformly
            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]))
            batch = data[batch_idx]

            self.train_step(batch, batch_size)

            if (n_batch + 1) % 500 == 0:
                y = self.generator(self.fixed_noise).numpy().squeeze()
                scores = []
                scores.append(np.linalg.norm(self.acf_real - acf(y.T, 250).mean(axis=1, keepdims=True)[:-1]))
                scores.append(np.linalg.norm(self.abs_acf_real - acf(y.T**2, 250).mean(axis=1, keepdims=True)[:-1]))
                scores.append(np.linalg.norm(self.le_real - acf(y.T, 250, le=True).mean(axis=1, keepdims=True)[:-1]))
                
                wass_avg = 0
                for i in range(len(y)):
                    wass_dist = wasserstein_distance(y[i, :], real_dist.transpose()[0])
                    wass_avg += wass_dist
                wass_avg /= len(y)
                
                scores.append(wass_avg)
                
                print("\nacf: {:.4f}, acf_abs: {:.4f}, le: {:.4f}, wass_dist: {:.4f}".format(*scores))


            if return_divergence:
                y = self.generator(self.fixed_noise).numpy().squeeze()
                wass_avg = 0
                for i in range(len(y)):
                    wass_avg += wasserstein_distance(y[i,126:], data[:,0,1,].transpose()[0])
                wass_avg /= len(y)
        
                train_divergence.append(wass_avg)

            progress.update(n_batch + 1)
            
        return train_divergence

    @tf.function
    def train_step(self, data, batch_size):

        noise = tf.random.normal([batch_size, *self.noise_shape])
        generated_data = self.generator(noise, training=False)

        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, *self.noise_shape])
        generated_data = self.generator(noise, training=False)
        
        noise = tf.random.normal([batch_size, *self.noise_shape])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=False)
            gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
