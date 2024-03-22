from preprocess.acf import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os, shutil

from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.utils import Progbar
from keras.models import load_model, Model
from keras.layers import Input, Concatenate
from tensorflow import convert_to_tensor
from math import floor, ceil
from scipy.stats import wasserstein_distance, entropy, beta

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

    def __init__(self, discriminator, generator, training_input, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, from_logits=True, log_returns = None, log_returns_preprocessed = None, scalers: dict=None):
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
            scalers (dict, optional): 3 data scalers used in data preprocessing
        """
        self.alpha_d = 1
        self.alpha_g = 1
        
        self.log_returns = log_returns
        self.log_returns_preprocessed = log_returns_preprocessed
        self.scalers = scalers
        self.discriminator = discriminator
        self.generator = generator
        self.optimal_generator = generator
        self.optimal_generator_batch = None
        self.train_post_divergence = []
        self.train_pre_divergence = []
        
        self.noise_shape = [self.generator.input_shape[1], training_input, self.generator.input_shape[-1]]

        self.loss = BinaryCrossentropy(from_logits=from_logits)

        self.generator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        
        self.file_name = "SP500_daily"
        self.figure_path = "figures/"
        self.file_path = "data/"+self.file_name+".csv"
        self.generator_path = ""
        self.retrain_path = "retrained_capstone/"
        
        self.bestPerformance = 1
        self.bestPerformanceBatch = -1



    def train(self, data, batch_size, n_batches):
        """training function of a GAN instance.
        Args:
            data (4d array): Training data in the following shape: (samples, timesteps, 1).
            batch_size (int): Batch size used during training.
            n_batches (int): Number of update steps taken.
        """ 
        self.n_batches = n_batches
        self.batchSize = batch_size
        progress = Progbar(n_batches)
        post_min_divergence = float('inf')
        pre_min_divergence = float('inf')
        
        for n_batch in range(n_batches):
            
            #create discrete beta distribution over number of rolling windows
            probabilities = beta.pdf(np.linspace(0.05, 0.95, data.shape[0]), 0.9, 0.9)
            probabilities /= np.sum(probabilities)
            
            # Sample windows from the discrete beta disribution
            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]), p = probabilities)
            batch = data[batch_idx]

            self.train_step(batch, batch_size)

            if (n_batch + 1) % 500 == 0:
                y = self.generator(self.fixed_noise).numpy().squeeze()
                scores = []
                scores.append(np.linalg.norm(self.acf_real - acf(y.T, 250).mean(axis=1, keepdims=True)[:-1]))
                scores.append(np.linalg.norm(self.abs_acf_real - acf(y.T**2, 250).mean(axis=1, keepdims=True)[:-1]))
                scores.append(np.linalg.norm(self.le_real - acf(y.T, 250, le=True).mean(axis=1, keepdims=True)[:-1]))
                
                # wass_avg = 0
                # for i in range(len(y)):
                #     wass_dist = wasserstein_distance(y[i, :], real_dist.transpose()[0])
                #     wass_avg += wass_dist
                # wass_avg /= len(y)
                
                #scores.append(self.train_divergence[-1])
                
                print("\nacf: {:.4f}, acf_abs: {:.4f}, le: {:.4f}".format(*scores))

            y = self.generateReturns(postprocessed = True)
            y_pre = self.generateReturns(postprocessed = False)
            
            post_wass_avg = 0
            pre_wass_avg = 0
            for i in range(len(y)):
                post_wass_avg += wasserstein_distance(y[i, :], self.log_returns)
                pre_wass_avg += wasserstein_distance(y_pre[i, :], self.log_returns_preprocessed)
            post_wass_avg /= len(y)
            pre_wass_avg /=len(y)
            
            if math.isnan(pre_wass_avg):
                break
            
            if post_wass_avg < post_min_divergence:
                self.post_optimal_generator = self.generator
                self.post_optimal_generator_batch = n_batch
                post_min_divergence = post_wass_avg
                
            if pre_wass_avg < pre_min_divergence:
                self.pre_optimal_generator = self.generator
                self.pre_optimal_generator_batch = n_batch
                pre_min_divergence = pre_wass_avg
                
            self.train_post_divergence.append(post_wass_avg)
            self.train_pre_divergence.append(pre_wass_avg)
            
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

    def savePlots(self):
        self.saveDivergencePlot()
        # self.saveLogReturnPlot()
        # self.saveLogReturnVsRealPlot()
        # self.saveHistPlot()
        # self.saveACFPlot()

    def saveDivergencePlot(self, preprocessed = True, postprocessed = True):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (20,10))

        ax[0].plot(self.train_post_divergence)
        postMinDiv = min(self.train_post_divergence)
        postMinDivIndex = self.train_post_divergence.index(postMinDiv)
        
        ax[1].plot(self.train_pre_divergence)
        preMinDiv = min(self.train_pre_divergence)
        preMinDivIndex = self.train_pre_divergence.index(preMinDiv)
        
        fig.suptitle('Wasserstein Distance over Training Iterations', fontsize=20)
        ax[0].set_title('Postprocessed Data Wasserstein Distance', fontsize=16)
        ax[0].set_xlabel('Training Iteration', fontsize=16)
        ax[0].set_ylabel('Wasserstein Divergence', fontsize=16)

        ax[1].set_title('Preprocessed Data Wasserstein Distance', fontsize=16)
        ax[1].set_xlabel('Training Iteration', fontsize=16)
        ax[1].set_ylabel('Wasserstein Divergence', fontsize=16)
        
        preText= "Epoch={:.0f}, Divergence Score={:.5f}".format(preMinDivIndex, preMinDiv)
        postText= "Epoch={:.0f}, Divergence Score={:.5f}".format(postMinDivIndex, postMinDiv)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        
        bbox_props1 = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops1=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
        kw1 = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops1, bbox=bbox_props1, ha="right", va="top")
        
        ax[0].annotate(postText, xy=(postMinDivIndex, postMinDiv), xytext=(0.94,0.96), **kw)
        ax[1].annotate(preText, xy=(preMinDivIndex, preMinDiv), xytext=(0.94,0.96), **kw1)
        
        plt.savefig(f"{self.figure_path}Wass_Dist_{self.file_name}_Alpha_D_{self.alpha_d}_Alpha_G_{self.alpha_g}_BatchSize_{self.batchSize}.png")
        
    # def saveLogReturnPlot(self):
        
    # def saveLogReturnVsRealPlot(self):
        
    # def saveHistPlot(self):
        
    # def saveACFPlot(self):

    
    def generateReturns(self, postprocessed = False):
        y = self.generator(self.fixed_noise).numpy().squeeze()

        if not postprocessed:
            return y
        
        y = (y - y.mean(axis=0))/y.std(axis=0)
        y = self.scalers.get('standardScaler2').inverse_transform(y)
        y = np.array([self.scalers.get('gaussianize').inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
        y = self.scalers.get('standardScaler1').inverse_transform(y)

        # some basic filtering to reduce the tendency of GAN to produce extreme returns
        # y -= y.mean()
        return y