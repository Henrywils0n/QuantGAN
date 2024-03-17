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
receptive_field_size = 127  # p. 17
log_returns_rolled = rolling_window(log_returns_preprocessed, receptive_field_size)
data_size = log_returns.shape[0]

# Noise
beta = 1.5
use_gen_noise = False

generalized_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[512, 1, data_size + receptive_field_size - 1, 3]))
gaussian_noise = normal([512, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])

# Training

train = False

if train:
	gan = GAN(discriminator, generator, 2 * receptive_field_size - 1, lr_d=1e-4, lr_g=3e-5)
	gan.acf_real = acf(log_returns_preprocessed, 250)[:-1]
	gan.abs_acf_real = acf(log_returns_preprocessed**2, 250)[:-1]
	gan.le_real = acf(log_returns_preprocessed, 250, le=True)[:-1]
	gan.fixed_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[128, 1, data_size + receptive_field_size - 1, 3])) if use_gen_noise \
		else normal([128, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])
	data = np.expand_dims(np.moveaxis(log_returns_rolled, 0,1), 1).astype('float32')
	batch_size = 128
	n_batches = 5000
	gan.train(data, batch_size, n_batches, log_returns)
	generator.save(f'trained_generator_{file_name}_Alpha_D_{gan.alpha_d}_Alpha_G_{gan.alpha_g}_BatchSize_{batch_size}')
	#zip -r trained_generator_{file_name}.zip trained_generator_{file_name}/
	#files.download(f'trained_generator_{file_name}.zip')
else:
	print(f"Loading: {generator_path}trained_generator_{file_name}")
	generator = load_model(f"./trained_models_capstone/batchsize 1000/{generator_path}trained_generator_{file_name}_Alpha_D_5.0_Alpha_G_5.0_BatchSize_1000")
	# generator = load_model(f"/temporalCN/trained/trained_generator_ShanghaiSE_daily")

# Generate
noise = normal([512, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])
y = generator(noise).numpy().squeeze()
y = (y - y.mean(axis=0))/y.std(axis=0)
y = standardScaler2.inverse_transform(y)
y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
y = standardScaler1.inverse_transform(y)

# some basic filtering to redue the tendency of GAN to produce extreme returns
y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]
# print(y)
# y -= y.mean()

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
print(avg_divergence)