import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import argparse



from scipy.stats import gennorm
from scipy.spatial import distance
from preprocess.acf import *
from preprocess.gaussianize import *

import tensorflow as tf
from scipy.stats import wasserstein_distance

physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

# Specify which GPU to use
gpu_index = 1  # Index of the GPU to use
tf.config.set_visible_devices(physical_devices[gpu_index], 'GPU')

# Verify that TensorFlow is using the desired GPU
print("Selected GPU:", tf.config.get_visible_devices('GPU')[0])

from tensorflow import convert_to_tensor
from tensorflow.random import normal
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#from google.colab import files
from model.tf_gan import GAN
from model.tf_tcn import *

## Set all flags before running

# for testing code
testing = False

# whether you want to save plots
save_plots = True

# training
train = True

# Gen_Gaus
use_gen_noise = False

parser = argparse.ArgumentParser()

# Add argument for alpha_d
parser.add_argument("--alpha_d", type=float, help="Value for alpha_d")

# Add argument for alpha_g
parser.add_argument("--alpha_g", type=float, help="Value for alpha_g")

# Parse the command line arguments
args = parser.parse_args()

# Access the values of alpha_d and alpha_g
alpha_d = args.alpha_d
alpha_g = args.alpha_g

# Print out the values of alpha_d and alpha_g
print("Value of alpha_d:", alpha_d)
print("Value of alpha_g:", alpha_g)

# Data
file_name = "SP500_daily"
file_path = "data/"+file_name+".csv"
generator_path = ""
metrics_path = "metrics/"

if testing:
	figure_path = "test_training_figures/"
	model_path = "test_training_models/"
	metrics_file = "test_metrics.csv"
else:
	figure_path = "redo_figures/"
	model_path = "redo_capstone/"
	metrics_file = "redo_official_metrics.csv"


def generateReturns(generator, noise, postprocessed = False):
	y = generator(noise).numpy().squeeze()
	if not postprocessed:
		return y
	y = (y - y.mean(axis=0))/y.std(axis=0)
	y = standardScaler2.inverse_transform(y)
	y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
	y = standardScaler1.inverse_transform(y)

	# some basic filtering to redue the tendency of GAN to produce extreme returns
	#y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]

	return y

def dateparse(d):
	return pd.Timestamp(d)

# Load Data
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

# alpha_d_list = [0.5, 1, 3, 5, 10, 100]
# alpha_g_list = [0.5, 1, 3, 5, 10, 100]

# valid_combinations = []
# trained_combinations = [(3,3), (5,3), (5,5), (10, 5), (10, 10)]

# # Iterate through each combination
# for alpha_d in alpha_d_list:
#     for alpha_g in alpha_g_list:
#     # Check the conditions for validity
#         if alpha_d <= 1 and alpha_g >= alpha_d / (alpha_d + 1) and (alpha_d, alpha_g) not in trained_combinations:
#             valid_combinations.append((alpha_d, alpha_g))
#         elif alpha_d > 1 and alpha_g >= alpha_d / 2 and alpha_g <= alpha_d and (alpha_d, alpha_g) not in trained_combinations:
#             valid_combinations.append((alpha_d, alpha_g))
# print(valid_combinations)
batch_size = 1000
n_batches = 1000

pre_acf_real = acf(log_returns_preprocessed, 250)[:-1]
pre_squared_acf_real = acf(log_returns_preprocessed**2, 250)[:-1]
pre_le_real = acf(log_returns_preprocessed, 250, le=True)[:-1]

acf_real = acf(log_returns, 250)[:-1]
squared_acf_real = acf(log_returns**2, 250)[:-1]
le_real = acf(log_returns, 250, le=True)[:-1]

beta = 1.5


# Noise
generalized_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[512, 1, data_size + receptive_field_size - 1, 3]))
gaussian_noise = normal([512, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])

# Training
if train:
	gan = GAN(discriminator, generator, 2 * receptive_field_size - 1, lr_d=1e-4, lr_g=3e-5, log_returns=log_returns[:,0], log_returns_preprocessed=log_returns_preprocessed[:,0], scalers=scalers, testing=testing)
	# Define ModelCheckpoint callbacks for saving the generator models
	gan.alpha_d = alpha_d
	gan.alpha_g = alpha_g
	gan.acf_real = pre_acf_real
	gan.squared_acf_real = pre_squared_acf_real
	gan.le_real = pre_le_real
	gan.fixed_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[128, 1, data_size + receptive_field_size - 1, 3])) if use_gen_noise \
		else normal([128, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])
	data = np.expand_dims(np.moveaxis(log_returns_rolled, 0,1), 1).astype('float32')
	# Get the current time
	current_time = datetime.datetime.now()
	# Format the current time as a string
	time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

	print(time_string + ": Training, Alpha_d:" + str(alpha_d) + "Alpha_G:" + str(alpha_g) +  "...")
	gan.train(data, batch_size, n_batches)
	gan.post_optimal_generator.save(f'{model_path}trained_post_generator_Batch_{gan.post_optimal_generator_batch}_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}')
	gan.pre_optimal_generator.save(f'{model_path}trained_pre_generator_Batch_{gan.pre_optimal_generator_batch}_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}')
	# check if generator diverged, if not save the final epoch
	y = generateReturns(generator, gaussian_noise, postprocessed=True)
	if not np.isnan(y).all():
		generator.save(f'{model_path}trained_generator_final_epoch_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}')
	gan.savePlots()
	#load in optimal generator
	#generator = load_model(f"{model_path}trained_pre_generator_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}")
	#zip -r trained_generator_{file_name}.zip trained_generator_{file_name}/
	#files.download(f'trained_generator_{file_name}.zip')
	# plt.plot(train_divergence)
	# plt.title("Wasserstein Distance over Training Iterations")
	# plt.xlabel('Training Iteration')
	# plt.ylabel('Wasserstein Distance')
	# plt.grid(axis = 'y')
	# plt.savefig(f"{figure_path}Wass_Dist_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}.png")
else:
	print(f"Loading: {model_path}trained_pre_generator_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}")
	#generator = load_model(f"{model_path}trained_pre_generator_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}")
	generator = load_model(f"{model_path}trained_post_generator_Batch_249_SP500_daily_Alpha_D_10_Alpha_G_10_BatchSize_1000_N_Batches_3000/")
	# generator = load_model(f"/temporalCN/trained/trained_generator_ShanghaiSE_daily")

# # Generate
pre_y = generateReturns(gan.pre_optimal_generator, gaussian_noise, postprocessed=True)
post_y = generateReturns(gan.post_optimal_generator, gaussian_noise, postprocessed=True)
final_y = generateReturns(generator, gaussian_noise, postprocessed=True)

returns_dict = { "pre" : pre_y, "post" : post_y}

if not np.isnan(final_y).all():	
	returns_dict["final"] = final_y


for prefix, y in returns_dict.items():

	if save_plots:
		# # Plot Paths and Avg
		ySum = y[:100].cumsum(axis=1).mean(axis=0)

		fig, ax = plt.subplots(figsize=(16,9))
		ax.plot(ySum, alpha=1, lw = 4, zorder = 2)
		ax.plot(np.cumsum(y[0:50], axis=1).T, alpha=0.65, zorder = 1)
		ax.legend(['Average'])

		ax.set_title('Generated Log Return Paths and Their Average'.format(len(y)), fontsize=20)
		ax.set_xlabel('Days', fontsize=16)
		ax.set_ylabel('Cumalative Log Return', fontsize=16)

		fig.savefig(f"{figure_path}{prefix}_Log_Return_Paths_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}.png")
		plt.close()
		# # Avg Vs True
		fig, ax = plt.subplots(figsize=(16,9))
		ax.plot(np.cumsum(log_returns, axis=0))
		ax.set_title('Generated log Return Paths Compared to Real Return Path'.format(len(y)), fontsize=20)
		ax.set_xlabel('Days', fontsize=16)
		ax.set_ylabel('Cumalative Log Return', fontsize=16)

		ax.plot(ySum, alpha=1, lw = 3)

		ax.legend(['Historical returns', 'Average of Synthetic Returns'])

		fig.savefig(f"{figure_path}{prefix}_Avg_Vs_True_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}.png")
		plt.close()

		# # Histograms
		n_bins = 50
		windows = [1, 5, 20, 100]

		fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

		for i in range(len(windows)):
			row = min(max(0, i-1), 1)
			col = i % 2
			real_dist = rolling_window(log_returns, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
			fake_dist = rolling_window(y.T[1][:], windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
			axs[row, col].hist(np.array([real_dist, fake_dist], dtype='object'), bins=50, density=True)
			axs[row,col].set_xlim(*np.quantile(fake_dist, [0.001, .999]))
			
			axs[row,col].set_title('{} day return distribution'.format(windows[i]), size=16)
			axs[row,col].yaxis.grid(True, alpha=0.5)
			axs[row,col].set_xlabel('Cumalative log return')
			axs[row,col].set_ylabel('Frequency')

		axs[0,0].legend(['Historical returns', 'Synthetic returns'])

		fig.savefig(f"{figure_path}{prefix}_Histograms_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}.png")
		plt.close()

		# # ACF scores
		fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

		axs[0,0].plot(acf(log_returns, 100))
		axs[0,0].plot(acf(y.T, 100).mean(axis=1))
		axs[0,0].set_ylim(-0.1, 0.1)
		axs[0,0].set_title('Identity log returns')
		axs[0,1].plot(acf(log_returns**2, 100))
		axs[0,1].set_ylim(-0.05, 0.5)
		axs[0,1].plot(acf(y.T**2, 100).mean(axis=1))
		axs[0,1].set_title('Squared log returns')
		axs[1,0].plot(abs(acf(log_returns, 100, le=True)))
		axs[1,0].plot(abs(acf(y.T, 100, le=True).mean(axis=1)))
		axs[1,0].set_ylim(-0.05, 0.4)
		axs[1,0].set_title('Absolute')
		axs[1,1].plot(acf(log_returns, 100, le=True))
		axs[1,1].plot(acf(y.T, 100, le=True).mean(axis=1))
		axs[1,1].set_ylim(-0.2, 0.1)
		axs[1,1].set_title('Leverage effect')
		axs[0,0].legend(['Historical returns', 'Synthetic returns'])

		for ax in axs.flat: 
			ax.grid(True)
			ax.axhline(y=0, color='k')
			ax.axvline(x=0, color='k')
		plt.setp(axs, xlabel='Lag (number of days)')

		fig.savefig(f"{figure_path}{prefix}_ACF_{file_name}_Alpha_D_{alpha_d}_Alpha_G_{alpha_g}_BatchSize_{batch_size}_N_Batches_{n_batches}.png")
		plt.close()
	# ACF Scores
	id_acf_score = np.linalg.norm(acf_real - acf(y.T, 250).mean(axis=1, keepdims=True)[:-1])
	squared_acf_score = np.linalg.norm(squared_acf_real - acf(y.T**2, 250).mean(axis=1, keepdims=True)[:-1])
	le_score = np.linalg.norm(le_real - acf(y.T, 250, le=True).mean(axis=1, keepdims=True)[:-1])

	# Wass_Dist
	wass_avg = 0
	for i in range(len(y)):
		wass_avg += wasserstein_distance(y[i, :], log_returns[:,0])
	wass_avg /= len(y)

	# # JS-Divergence
	r_data = []
	for data_point in log_returns:
		r_data.append(data_point[0])
	real_data = np.array(r_data)
	real_histogram = np.histogram(real_data, bins=int(len(real_data)/10), density=True)

	divergences = []

	for return_path in y:
		hist = np.histogram(return_path, bins=int(len(real_data)/10), density=True)
		divergences.append(distance.jensenshannon(real_histogram[0], hist[0], 2.0))

	avg_divergence = np.mean(np.array(divergences))

	# Save Scores to CSV
	metrics = {
		'alpha_d': [alpha_d],
		'alpha_g': [alpha_g],
		'batch_size': [batch_size],
		'num_batches': [n_batches],
		'id_acf_score': [id_acf_score],
		'squared_acf_score': [squared_acf_score],
		'le_score': [le_score],
		'wass_avg': [wass_avg],
		'avg_divergence': [avg_divergence]
	}


	metrics_file_path = f'{metrics_path}{prefix}_{metrics_file}'

	# Create or load the DataFrame
	try:
		metrics_df = pd.read_csv(metrics_file_path)
	except FileNotFoundError:
		metrics_df = pd.DataFrame(columns=['alpha_d', 'alpha_g', 'batch_size', 'num_batches', 'id_acf_score', 'squared_acf_score', 'le_score', 'wass_avg', 'avg_divergence'])

	metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)], ignore_index=True)

	metrics_df.to_csv(metrics_file_path, index=False)