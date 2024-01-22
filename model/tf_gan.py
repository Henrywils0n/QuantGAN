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

#Alpha GAN generator and discriminator loss functions from Justin Veiner, github.com/justin-veiner/MASc
#from alpha_loss import dis_loss_alpha, gen_loss_alpha
def dis_loss_alpha(self, fake_predicted_labels, real_predicted_labels, alpha_d = 3.0, gp = False, gp_coef = 5.0):
    """
    fake_predicted_labels: fake predicted values
    real_predicted_labels: real predicted values
    alpha_d: alpha parameter for the discriminator loss function (positive float, default 3.0)
    gp: add a gradient penalty to the discriminator's loss function (bool, default False)
    gp_coef: coefficient for gradient penalty term if added (float, default 5.0)
    img: I DO NOT KNOW WHAT THIS IS. Commented out this section. Think has something to do with the real training data.
    """
    real_expr = tf.math.pow(real_predicted_labels, ((alpha_d-1)/alpha_d)*tf.ones_like(real_predicted_labels))
    real_loss = tf.math.reduce_mean(real_expr)
    fake_expr = tf.math.pow(1 - fake_predicted_labels, ((alpha_d-1)/alpha_d)*tf.ones_like(fake_predicted_labels))
    fake_loss = tf.math.reduce_mean(fake_expr)
    r1_penalty = 0
    """
    if gp:
        gradients = tf.gradients(-tf.math.log(1 / real_predicted_labels - 1), [self.img])[0]
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    """
    loss_expr = -(alpha_d/(alpha_d - 1))*(real_loss + fake_loss - 2.0)


    return loss_expr + gp_coef*r1_penalty

def gen_loss_alpha(self, fake_predicted_labels, alpha_g = 3.0,l1 = False):
    """
    fake_predicted_labels: fake predicted values
    alpha_g: alpha parameter for the generator loss function (positive float, default 3.0)
    l1: I DO NOT KNOW WHAT THIS IS. Set to False by default. Liekly somethign to do with L1
    """
    fake_expr = tf.math.pow(1 - fake_predicted_labels, ((alpha_g-1)/alpha_g)*tf.ones_like(fake_predicted_labels))
    fake_loss = tf.math.reduce_mean(fake_expr)
    loss_expr = (alpha_g/(alpha_g - 1))*(fake_loss - 2.0)
    if l1:
        equil_val = (alpha_g)/(alpha_g - 1)*(tf.math.pow(2.0, 1/alpha_g) - 2)
        loss_expr = tf.math.abs(loss_expr - equil_val)
    return loss_expr

class GAN:
    """ Generative adverserial network class.

    Training code for a standard DCGAN using the Adam optimizer.
    Code taken in part from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
    """    

    def discriminator_loss(self, real_output, fake_output):
        real_loss = dis_loss_alpha(tf.ones_like(real_output), real_output)
        fake_loss = dis_loss_alpha(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return gen_loss_alpha(tf.ones_like(fake_output), fake_output)

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
        self.discriminator = discriminator
        self.generator = generator
        self.noise_shape = [self.generator.input_shape[1], training_input, self.generator.input_shape[-1]]

        #self.loss = BinaryCrossentropy(from_logits=from_logits)

        self.generator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def train(self, data, batch_size, n_batches):
        """training function of a GAN instance.
        Args:
            data (4d array): Training data in the following shape: (samples, timesteps, 1).
            batch_size (int): Batch size used during training.
            n_batches (int): Number of update steps taken.
        """ 
        progress = Progbar(n_batches)

        for n_batch in range(n_batches):
            # sample uniformly
            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]))
            batch = data[batch_idx]

            self.train_step(batch, batch_size)

            if (n_batch + 1) % 500 == 0:
              y = self.generator(self.fixed_noise).numpy().squeeze()
              scores = []
              scores.append(np.linalg.norm(self.acf_real - acf(y.T, 250).mean(axis=1, keepdims=True)))
              scores.append(np.linalg.norm(self.abs_acf_real - acf(y.T**2, 250).mean(axis=1, keepdims=True)))
              scores.append(np.linalg.norm(self.le_real - acf(y.T, 250, le=True).mean(axis=1, keepdims=True)))
              print("\nacf: {:.4f}, acf_abs: {:.4f}, le: {:.4f}".format(*scores))

            progress.update(n_batch + 1)

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
