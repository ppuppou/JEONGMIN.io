import pandas as pd 
import numpy as np 
#import janitor
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

ex1 = pd.read_csv('./data/ex1.csv')
plot_acf(ex1.x)
plt.show();


dat = pd.read_csv('./data/elecstock.csv')
dat.plot()
plt.show();

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat['value'], ax = ax1)
plot_pacf(dat['value'], ax = ax2, method = 'ywm')
plt.show();

dat_diff = dat.diff()
dat_diff.plot()
plt.show();

dat_diff.dropna(inplace = True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat_diff['value'], ax = ax1)
plot_pacf(dat_diff['value'], ax = ax2, method = 'ywm')
plt.show();


dat = pd.read_csv('./data/ex6.csv')
dat.plot()
plt.show();


from statsmodels.tsa.stattools import kpss
#from statsmodels.tsa.stattools import adfuller
# def kpss_test(series):    
#     statistic, p_value, n_lags, critical_values = kpss(series)
#     print(f'KPSS Statistic: {statistic}')
#     print(f'p-value: {p_value}')
#     print(f'num lags: {n_lags}')
#     print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
#print(adfuller(dat)[1])
print(kpss(dat)[1])

dat_diff = dat.diff()
dat_diff.plot()
plt.show();

dat_diff.dropna(inplace = True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat_diff['x'], ax = ax1)
plot_pacf(dat_diff['x'], ax = ax2, method = 'ywm')
plt.show();

dat_diff2 = dat.diff().diff()
dat_diff2.dropna(inplace = True)
dat_diff2.plot()
plt.show();

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat_diff2['x'], ax = ax1)
plot_pacf(dat_diff2['x'], ax = ax2, method = 'ywm')
plt.show();

import pmdarima as pm
model2 = pm.auto_arima(dat['x'], 
                   start_p=0, 
                   start_q=0,
                   max_p=5, 
                   max_q=5,
                   m=1,             
                   d=1,          
                   seasonal=False,   
                   start_P=0, 
                   D=None, 
                   trace=True,
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True)







dat = pd.read_csv('./data/depart.csv')
dat = dat.rename(columns = {'index' : 'date'})
dat['date'] = pd.to_datetime(dat['date'], format = '%Y %b')

import datetime
dat['date'] = dat['date'].dt.strftime('%Y-%m')
dat.index = dat['date']
dat = dat.drop(['date'], axis = 1)

dat.plot()
plt.show();

dat['value'] = np.log(dat['value'])
dat.plot()
plt.show();

from statsmodels.tsa.stattools import kpss
print('test statistic: %f' % kpss(dat)[0])
print('p-value: %f' % kpss(dat)[1])

from pmdarima.arima.utils import nsdiffs

nsdiffs(dat,
            m=12, 
            max_D=12,
            test='ch')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat['value'], ax = ax1)
plot_pacf(dat['value'], ax = ax2, method = 'ywm')
plt.show();

dat_diff = dat.diff()
dat_diff.plot()
plt.show();


dat_diff.dropna(inplace = True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))
plot_acf(dat_diff['value'], ax = ax1)
plot_pacf(dat_diff['value'], ax = ax2, method = 'ywm')
plt.show();

print('test statistic: %f' % kpss(dat_diff)[0])
print('p-value: %f' % kpss(dat_diff)[1])

dat_diff2 = dat.diff(12)
dat_diff2.plot()
plt.show();

dat_diff3 = dat_diff.diff(12)
dat_diff3.plot()
plt.show();


dat_diff3.dropna(inplace = True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
plot_acf(dat_diff3['value'], ax = ax1, lags = 24)
plot_pacf(dat_diff3['value'].squeeze(), ax = ax2, method = 'ywm', lags = 22)
plt.show();


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(dat, order=(0,1,1), seasonal_order=(0, 1, 0, 12)).fit()
print(model.summary())


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
plot_acf(model.resid, ax = ax1)
model.resid.hist(ax = ax2)
plt.show();

sm.stats.acorr_ljungbox(model.resid, lags=[12])