import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import dates
import random
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams['axes.unicode_minus'] = False
rc('font', family='AppleGothic')

df = pd.read_csv("/Users/unixking/Desktop/sobu/재고_연별월별.csv", encoding= "euc-kr")
df = df.dropna()

df["연도"] = df["연도"].apply(lambda x: str(x)[0:4]+ "/" + str(x)[4:6])

df['연도'] = pd.to_datetime(df['연도'])


x_train = df['자동차부품']
x_train2 = df['반도체·디스플레이장비']

result = seasonal_decompose(x_train, period=1)
result.plot()
plt.show()
plt.clf()


result = seasonal_decompose(x_train2, period=1)
result.plot()
plt.show()
plt.clf()



y = df.set_index(['연도'])
y = y[["자동차부품"]]


model = pm.auto_arima(y = y        # 데이터
                      , d = 1            # 차분 차수, ndiffs 결과!
                      , start_p = 0 
                      , max_p = 3   
                      , start_q = 0 
                      , max_q = 3   
                      , m = 1       
                      , seasonal = True # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True
                      , trace=True
                      )

model = pm.auto_arima (y, d = 1, seasonal = False, trace = True)
model.fit(y)

print(model.summary())

model.plot_diagnostics(figsize=(16, 8))
plt.show()



y_test = pd.date_range("2022/06/01", "2023/06/01", freq= "MS")

y_predict = pd.DataFrame(y_predict,index = y_test,columns=['Prediction'])


model_arima = ARIMA(y, order=(0,1,1))
model_arima_fit = model_arima.fit(disp = -1)

pre = model_arima_fit.forecast(13)[0]
pre = pd.DataFrame(pre, index = y_test)
pre

pre.to_csv("/Users/unixking/Desktop/sobu/자동차.csv")

#--------------------------------------------------------------------------------------------

y = df.set_index(['연도'])
y = y[["반도체·디스플레이장비"]]


model = pm.auto_arima(y = y        # 데이터
                      , d = 1            # 차분 차수, ndiffs 결과!
                      , start_p = 0 
                      , max_p = 3   
                      , start_q = 0 
                      , max_q = 3   
                      , m = 1       
                      , seasonal = True # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True
                      , trace=True
                      )

model = pm.auto_arima (y, d = 1, seasonal = False, trace = True)
model.fit(y)

print(model.summary())

model.plot_diagnostics(figsize=(16, 8))
plt.show()



y_test = pd.date_range("2022/06/01", "2023/06/01", freq= "MS")



model_arima = ARIMA(y, order=(0,1,1))
model_arima_fit = model_arima.fit(disp = -1)

pre = model_arima_fit.forecast(13)[0]
pre = pd.DataFrame(pre, index = y_test)
pre

pre.to_csv("/Users/unixking/Desktop/sobu/반도체.csv")