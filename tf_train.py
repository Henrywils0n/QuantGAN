import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gennorm
from scipy.spatial import distance
from preprocess.acf import *
from preprocess.gaussianize import *

import tensorflow as tf

from tensorflow import convert_to_tensor
from tensorflow.random import normal
from tensorflow.keras.models import load_model
#from google.colab import files
from model.tf_gan import GAN
from model.tf_tcn import *

# Data
file_name = "SP500_daily"
file_path = "data/"+file_name+".csv"
generator_path = ""
figure_path = "figures/"
retrain_path = "retrained_capstone/"

def dateparse(d):
	return pd.Timestamp(d)

data = pd.read_csv(file_path, parse_dates={'datetime': ['Date']}, date_parser=dateparse)
df = data['Close']

# Preprocess
returns = df.shift(1)/df - 1
log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
standardScaler1 = StandardScaler()
standardScaler2 = StandardScaler()
gaussianize = Gaussianize()
log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))
scalers = {'standardScaler1': standardScaler1, 
           'gaussianize': gaussianize, 
           'standardScaler2': standardScaler2}
receptive_field_size = 127  # p. 17
log_returns_rolled = rolling_window(log_returns_preprocessed, receptive_field_size)
data_size = log_returns.shape[0]

# Noise
beta = 1.5
use_gen_noise = False

generalized_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[512, 1, data_size + receptive_field_size - 1, 3]))
gaussian_noise = normal([512, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])

# Training
train = True

if train:
	gan = GAN(discriminator, generator, 2 * receptive_field_size - 1, lr_d=1e-4, lr_g=3e-5, log_returns=log_returns[:,0] ,scalers=scalers)
	gan.alpha_d = 1
	gan.alpha_g = 1
	gan.acf_real = acf(log_returns_preprocessed, 250)[:-1]
	gan.abs_acf_real = acf(log_returns_preprocessed**2, 250)[:-1]
	gan.le_real = acf(log_returns_preprocessed, 250, le=True)[:-1]
	gan.fixed_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[128, 1, data_size + receptive_field_size - 1, 3])) if use_gen_noise \
		else normal([128, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])
	data = np.expand_dims(np.moveaxis(log_returns_rolled, 0,1), 1).astype('float32')
	batch_size = 128
	n_batches = 500
	gan.train(data, batch_size, n_batches, log_returns)
	generator.save(f'{retrain_path}trained_generator_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}')
	gan.saveDivergencePlot()
	#zip -r trained_generator_{file_name}.zip trained_generator_{file_name}/
	#files.download(f'trained_generator_{file_name}.zip')
	# plt.plot(train_divergence)
	# plt.title("Wasserstein Distance over Training Iterations")
	# plt.xlabel('Training Iteration')
	# plt.ylabel('Wasserstein Distance')
	# plt.grid(axis = 'y')
	# plt.savefig(f"{figure_path}Wass_Dist_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}.png")
else:
	print(f"Loading: {generator_path}trained_generator_{file_name}")
	generator = load_model(f"./trained_models_capstone/batchsize 1000/{generator_path}trained_generator_{file_name}_Alpha_D_5.0_Alpha_G_5.0_BatchSize_1000")
	# generator = load_model(f"/temporalCN/trained/trained_generator_ShanghaiSE_daily")

# # Generate
# y = generator(generalized_noise).numpy().squeeze() if use_gen_noise \
#     else generator(gaussian_noise).numpy().squeeze()

# y = (y - y.mean(axis=0))/y.std(axis=0)
# y = standardScaler2.inverse_transform(y)
# y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
# y = standardScaler1.inverse_transform(y)

# # some basic filtering to redue the tendency of GAN to produce extreme returns
# # y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]
# y -= y.mean()

# # Plot Paths
# ySum = y[:100].cumsum(axis=1).mean(axis=0)

# fig, ax = plt.subplots(figsize=(16,9))
# ax.plot(ySum, alpha=1, lw = 4, zorder = 2)
# ax.plot(np.cumsum(y[0:50], axis=1).T, alpha=0.65, zorder = 1)
# ax.legend(['Average'])

# ax.set_title('Generated Log Return Paths and Their Average'.format(len(y)), fontsize=20)
# ax.set_xlabel('Days', fontsize=16)
# ax.set_ylabel('Cumalative Log Return', fontsize=16)

# # Avg Vs True
# fig, ax = plt.subplots(figsize=(16,9))
# ax.plot(np.cumsum(log_returns, axis=0))
# ax.set_title('Generated log Return Paths Compared to Real Return Path'.format(len(y)), fontsize=20)
# ax.set_xlabel('Days', fontsize=16)
# ax.set_ylabel('Cumalative Log Return', fontsize=16)

# #for i in [105, 150, 200]:
# #    ax.plot(y[100:i].cumsum(axis=1).mean(axis=0))
# ax.plot(ySum, alpha=1, lw = 3)

# ax.legend(['Historical returns', 'Average of Synthetic Returns'])


# # Histograms
# n_bins = 50
# windows = [1, 5, 20, 100]

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))


# for i in range(len(windows)):
#     row = min(max(0, i-1), 1)
#     col = i % 2
#     real_dist = rolling_window(log_returns, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
#     fake_dist = rolling_window(y.T[1][:], windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
#     axs[row, col].hist(np.array([real_dist, fake_dist], dtype='object'), bins=50, density=True)
#     axs[row,col].set_xlim(*np.quantile(fake_dist, [0.001, .999]))
    
#     axs[row,col].set_title('{} day return distribution'.format(windows[i]), size=16)
#     axs[row,col].yaxis.grid(True, alpha=0.5)
#     axs[row,col].set_xlabel('Cumalative log return')
#     axs[row,col].set_ylabel('Frequency')

# axs[0,0].legend(['Historical returns', 'Synthetic returns'])


# # ACF scores
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

# axs[0,0].plot(acf(log_returns, 100))
# axs[0,0].plot(acf(y.T, 100).mean(axis=1))
# axs[0,0].set_ylim(-0.1, 0.1)
# axs[0,0].set_title('Identity log returns')
# axs[0,1].plot(acf(log_returns**2, 100))
# axs[0,1].set_ylim(-0.05, 0.5)
# axs[0,1].plot(acf(y.T**2, 100).mean(axis=1))
# axs[0,1].set_title('Squared log returns')
# axs[1,0].plot(abs(acf(log_returns, 100, le=True)))
# axs[1,0].plot(abs(acf(y.T, 100, le=True).mean(axis=1)))
# axs[1,0].set_ylim(-0.05, 0.4)
# axs[1,0].set_title('Absolute')
# axs[1,1].plot(acf(log_returns, 100, le=True))
# axs[1,1].plot(acf(y.T, 100, le=True).mean(axis=1))
# axs[1,1].set_ylim(-0.2, 0.1)
# axs[1,1].set_title('Leverage effect')
# axs[0,0].legend(['Historical returns', 'Synthetic returns'])

# for ax in axs.flat: 
#   ax.grid(True)
#   ax.axhline(y=0, color='k')
#   ax.axvline(x=0, color='k')
# plt.setp(axs, xlabel='Lag (number of days)')

# # JS-Divergence
# r_data = []
# for data_point in log_returns:
# 	r_data.append(data_point[0])
# real_data = np.array(r_data)
# real_histogram = np.histogram(real_data, bins=int(len(real_data)/10), density=True)

# divergences = []

# for return_path in y:
# 	hist = np.histogram(return_path, bins=int(len(real_data)/10), density=True)
# 	divergences.append(distance.jensenshannon(real_histogram[0], hist[0], 2.0))

# avg_divergence = np.mean(np.array(divergences))
# print(avg_divergence)