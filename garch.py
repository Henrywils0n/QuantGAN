from arch import arch_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocess.acf import *
from preprocess.gaussianize import *

def format_yahoo(file_path: str):
    df = pd.read_csv(file_path)
    dataset = df[["Date", "Close"]]
    dataset.to_csv(file_path, index=False)
    
file_name = "SP500_daily"
file_path = "data/"+file_name+".csv"
generator_path = ""

format_yahoo(file_path)

def dateparse(d):
    return pd.Timestamp(d)

data = pd.read_csv(file_path, parse_dates={'datetime': ['Date']}, date_parser=dateparse)
df = data['Close']
# confirm data loaded

returns = df.shift(1)/df - 1
log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
standardScaler1 = StandardScaler()
standardScaler2 = StandardScaler()
gaussianize = Gaussianize()
log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))
receptive_field_size = 127  # p. 17
log_returns_rolled = rolling_window(log_returns_preprocessed, receptive_field_size)
data_size = log_returns.shape[0]

model = arch_model(log_returns, vol='Garch', p=1, q=1)
result = model.fit()

volatilities = result.conditional_volatility

garch_returns = np.array(log_returns[0])

for i in range(len(volatilities)):
    curr_vol = volatilities[i]
    noise = np.random.normal(0, 1)
    garch_returns = np.append(garch_returns, curr_vol*noise)

plt.figure(figsize=(15, 6))
plt.plot(np.cumsum(log_returns, axis=0), label='Original Log Returns')
plt.plot(np.cumsum(garch_returns, axis=0), label='GARCH Model Synthetic Returns')
plt.legend()

plt.title('Generated Log Returns Vs True', fontsize=20)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Cumulative Log Return', fontsize=16)

plt.show() 

# Display model summary
# print(result.summary())

# # Plot volatility
# result.plot()

# plt.show()

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


# for ax in axs.flat: 
#   ax.grid(True)
#   ax.axhline(y=0, color='k')
#   ax.axvline(x=0, color='k')
# plt.setp(axs, xlabel='Lag (number of days')

