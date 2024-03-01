import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gennorm
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
strategy = tf.distribute.MirroredStrategy()

train = True

if train:
	gan = GAN(discriminator, generator, 2 * receptive_field_size - 1, lr_d=1e-4, lr_g=3e-5)
	gan.acf_real = acf(log_returns_preprocessed, 250)[:-1]
	gan.abs_acf_real = acf(log_returns_preprocessed**2, 250)[:-1]
	gan.le_real = acf(log_returns_preprocessed, 250, le=True)[:-1]
	gan.fixed_noise = convert_to_tensor(gennorm.rvs(beta=beta, size=[128, 1, data_size + receptive_field_size - 1, 3])) if use_gen_noise \
		else normal([128, 1, len(log_returns_preprocessed) + receptive_field_size - 1, 3])
	data = np.expand_dims(np.moveaxis(log_returns_rolled, 0,1), 1).astype('float32')
	batch_size = 1000
	n_batches = 3000
	gan.train(data, batch_size, n_batches)
	generator.save(f'trained_generator_{file_name}')
	#zip -r trained_generator_{file_name}.zip trained_generator_{file_name}/
	#files.download(f'trained_generator_{file_name}.zip')
else:
	print(f"Loading: {generator_path}trained_generator_{file_name}")
	generator = load_model(f"{generator_path}trained_generator_{file_name}")
	# generator = load_model(f"/temporalCN/trained/trained_generator_ShanghaiSE_daily")