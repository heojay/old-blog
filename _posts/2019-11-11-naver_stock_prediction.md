---
title: "네이버 주식 예측"
date: 2019-11-11 08:26:28 -0400
categories: data_science


---

주식 예측 Tutorial을 내 나름대로 재구성해보기로 한다.  
네이버 핵데이를 위한 선행학습이니 네이버의 주가를 예측해보도록 하자.

## 1. 주식 데이터 수집

주식 데이터는 네이버 금융 사이트에서 파싱하기로 하자. [출처](http://blog.quantylab.com/crawling_naverfin_daycandle.html)




```python
code = '035420' #네이버 주식 Code

import requests
url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
res = requests.get(url)
res.encoding = 'utf-8'
```


```python
from bs4 import BeautifulSoup
soap = BeautifulSoup(res.text, 'lxml')

el_table_navi = soap.find("table", class_="Nnavi")
el_td_last = el_table_navi.find("td", class_="pgRR")
pg_last = el_td_last.a.get('href').rsplit('&')[1]
pg_last = int(pg_last.split('=')[1])
```


```python
import traceback
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def parse_page(code, page):
    try:
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
        res = requests.get(url)
        _soap = BeautifulSoup(res.text, 'lxml')
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        return _df
    except Exception as e:
        traceback.print_exc()
    return None
```

데이터 분석의 용이성을 위해 네이버의 액면분할 날짜인 2018년 10월 중순 경부터 수집하기로 한다.


```python
import datetime
str_datefrom = datetime.datetime.strftime(datetime.datetime(year=2018, month=10, day=15), '%Y.%m.%d')
str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')
```


```python
df = None
for page in range(1, pg_last+1):
    _df = parse_page(code, page)
    _df_filtered = _df[_df['날짜'] > str_datefrom]
    if df is None:
        df = _df_filtered
    else:
        df = pd.concat([df, _df_filtered])
    if len(_df) > len(_df_filtered):
        break
```


```python
df.to_csv("naver.csv", mode='w') # 저장
```


```python
df = pd.read_csv("naver.csv", index_col=0) # 불러오기
```

## 2. 주식 데이터 정리

이렇게 수집한 데이터를 어떻게 분석할 수 있을 지 알아보자.

### a. 날짜로 index 설정하기  
우선 날짜 column을 datetime 자료형으로 바꾸는 것부터 시작한다.


```python
df['날짜'] = df['날짜'].apply(pd.to_datetime)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 265 entries, 1 to 5
    Data columns (total 7 columns):
    날짜     265 non-null datetime64[ns]
    종가     265 non-null float64
    전일비    265 non-null float64
    시가     265 non-null float64
    고가     265 non-null float64
    저가     265 non-null float64
    거래량    265 non-null float64
    dtypes: datetime64[ns](1), float64(6)
    memory usage: 16.6 KB


이렇게 바꾼 날짜 column을 index로 설정하자


```python
df.set_index('날짜', inplace = True)
df = df.sort_index()
```

groupby 연산으로 연도별, 날짜별로 쉽게 묶어 볼 수도 있다.


```python
#연도-월 별 평균
df.groupby([df.index.year, df.index.month]).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
    <tr>
      <th>날짜</th>
      <th>날짜</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">2018</th>
      <th>10</th>
      <td>122000.000000</td>
      <td>3541.666667</td>
      <td>123083.333333</td>
      <td>124833.333333</td>
      <td>118833.333333</td>
      <td>1.281776e+06</td>
    </tr>
    <tr>
      <th>11</th>
      <td>116431.818182</td>
      <td>2590.909091</td>
      <td>116204.545455</td>
      <td>118909.090909</td>
      <td>114250.000000</td>
      <td>5.978032e+05</td>
    </tr>
    <tr>
      <th>12</th>
      <td>122947.368421</td>
      <td>2184.210526</td>
      <td>123421.052632</td>
      <td>125105.263158</td>
      <td>120868.421053</td>
      <td>4.246506e+05</td>
    </tr>
    <tr>
      <th rowspan="11" valign="top">2019</th>
      <th>1</th>
      <td>130931.818182</td>
      <td>2636.363636</td>
      <td>130477.272727</td>
      <td>133454.545455</td>
      <td>128477.272727</td>
      <td>4.282800e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>127882.352941</td>
      <td>2000.000000</td>
      <td>128176.470588</td>
      <td>130735.294118</td>
      <td>126294.117647</td>
      <td>5.233395e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130300.000000</td>
      <td>1750.000000</td>
      <td>130825.000000</td>
      <td>132325.000000</td>
      <td>128650.000000</td>
      <td>3.986966e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120636.363636</td>
      <td>840.909091</td>
      <td>120909.090909</td>
      <td>121795.454545</td>
      <td>119704.545455</td>
      <td>5.962936e+05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>117333.333333</td>
      <td>1714.285714</td>
      <td>117357.142857</td>
      <td>119428.571429</td>
      <td>115761.904762</td>
      <td>5.516289e+05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>112605.263158</td>
      <td>1394.736842</td>
      <td>112526.315789</td>
      <td>113789.473684</td>
      <td>110973.684211</td>
      <td>4.415339e+05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>122891.304348</td>
      <td>2000.000000</td>
      <td>122239.130435</td>
      <td>124347.826087</td>
      <td>121021.739130</td>
      <td>4.571633e+05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>142714.285714</td>
      <td>2071.428571</td>
      <td>142119.047619</td>
      <td>144619.047619</td>
      <td>140095.238095</td>
      <td>4.696588e+05</td>
    </tr>
    <tr>
      <th>9</th>
      <td>154315.789474</td>
      <td>1657.894737</td>
      <td>153868.421053</td>
      <td>155710.526316</td>
      <td>152131.578947</td>
      <td>3.596366e+05</td>
    </tr>
    <tr>
      <th>10</th>
      <td>154357.142857</td>
      <td>2619.047619</td>
      <td>153952.380952</td>
      <td>156333.333333</td>
      <td>151928.571429</td>
      <td>3.454135e+05</td>
    </tr>
    <tr>
      <th>11</th>
      <td>163785.714286</td>
      <td>2142.857143</td>
      <td>163928.571429</td>
      <td>165785.714286</td>
      <td>161642.857143</td>
      <td>4.427626e+05</td>
    </tr>
  </tbody>
</table>
</div>



### b. Resampling  
resample이란, 시간 간격을 재조정하는 것으로, 구간이 작아지면 업샘플링(up-sampling), 구간이 커지면(down-sampling)이라고 한다.  

업샘플링의 경우, 없는 데이터를 만들어야 하기 때문에 이를 어떤 값으로 채워줄 지 결정해야 한다. ffill, bfill 중 하나를 고르면 된다.  

반면 다운샘플링의 경우, 원래의 데이터가 그룹으로 묶이기 때문에 groupby 처럼 그룹 연산을 해서 대표값을 구해야 한다.  

묶는 방법은 다음과 같다.

<table border="1" class="docutils">
<colgroup>
<col width="13%" />
<col width="87%" />
</colgroup>
<thead valign="bottom">
<tr class="row-odd"><th class="head">Alias</th>
<th class="head">Description</th>
</tr>
</thead>
<tbody valign="top">
<tr class="row-even"><td>B</td>
<td>business day frequency</td>
</tr>
<tr class="row-odd"><td>C</td>
<td>custom business day frequency (experimental)</td>
</tr>
<tr class="row-even"><td>D</td>
<td>calendar day frequency</td>
</tr>
<tr class="row-odd"><td>W</td>
<td>weekly frequency</td>
</tr>
<tr class="row-even"><td>M</td>
<td>month end frequency</td>
</tr>
<tr class="row-odd"><td>SM</td>
<td>semi-month end frequency (15th and end of month)</td>
</tr>
<tr class="row-even"><td>BM</td>
<td>business month end frequency</td>
</tr>
<tr class="row-odd"><td>CBM</td>
<td>custom business month end frequency</td>
</tr>
<tr class="row-even"><td>MS</td>
<td>month start frequency</td>
</tr>
<tr class="row-odd"><td>SMS</td>
<td>semi-month start frequency (1st and 15th)</td>
</tr>
<tr class="row-even"><td>BMS</td>
<td>business month start frequency</td>
</tr>
<tr class="row-odd"><td>CBMS</td>
<td>custom business month start frequency</td>
</tr>
<tr class="row-even"><td>Q</td>
<td>quarter end frequency</td>
</tr>
<tr class="row-odd"><td>BQ</td>
<td>business quarter endfrequency</td>
</tr>
<tr class="row-even"><td>QS</td>
<td>quarter start frequency</td>
</tr>
<tr class="row-odd"><td>BQS</td>
<td>business quarter start frequency</td>
</tr>
<tr class="row-even"><td>A</td>
<td>year end frequency</td>
</tr>
<tr class="row-odd"><td>BA</td>
<td>business year end frequency</td>
</tr>
<tr class="row-even"><td>AS</td>
<td>year start frequency</td>
</tr>
<tr class="row-odd"><td>BAS</td>
<td>business year start frequency</td>
</tr>
<tr class="row-even"><td>BH</td>
<td>business hour frequency</td>
</tr>
<tr class="row-odd"><td>H</td>
<td>hourly frequency</td>
</tr>
<tr class="row-even"><td>T, min</td>
<td>minutely frequency</td>
</tr>
<tr class="row-odd"><td>S</td>
<td>secondly frequency</td>
</tr>
<tr class="row-even"><td>L, ms</td>
<td>milliseconds</td>
</tr>
<tr class="row-odd"><td>U, us</td>
<td>microseconds</td>
</tr>
<tr class="row-even"><td>N</td>
<td>nanoseconds</td>
</tr>
</tbody>
</table>


```python
df[['종가']].resample('MS').mean().plot(kind='bar')
plt.title('네이버 월별 종가')
```




    Text(0.5, 1.0, '네이버 월별 종가')




![png](/assets/fig/2019-11-11/output_19_1.png)


### c. Rolling  

이동 평균(Moving Average)은 정해진 기간의 평균 가격을 말한다. 불규칙한 노이즈 같은 주가 변동을 필터링하여 일일 간의 동요가 줄어들기 때문에 유용하게 사용될 수 있다.


```python
df[['시가']].plot(figsize=(15,7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123008a90>




![png](/assets/fig/2019-11-11/output_21_1.png)



```python
df.rolling(7).mean().head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
    <tr>
      <th>날짜</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-10-16</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-17</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-19</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-22</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-23</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-10-24</th>
      <td>128214.285714</td>
      <td>2857.142857</td>
      <td>129500.000000</td>
      <td>131071.428571</td>
      <td>125785.714286</td>
      <td>1.385943e+06</td>
    </tr>
    <tr>
      <th>2018-10-25</th>
      <td>126142.857143</td>
      <td>3642.857143</td>
      <td>127071.428571</td>
      <td>128571.428571</td>
      <td>123500.000000</td>
      <td>1.522143e+06</td>
    </tr>
    <tr>
      <th>2018-10-26</th>
      <td>123714.285714</td>
      <td>4000.000000</td>
      <td>125142.857143</td>
      <td>126357.142857</td>
      <td>120714.285714</td>
      <td>9.849843e+05</td>
    </tr>
    <tr>
      <th>2018-10-29</th>
      <td>121500.000000</td>
      <td>3785.714286</td>
      <td>122928.571429</td>
      <td>124285.714286</td>
      <td>118642.857143</td>
      <td>9.520089e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['종가'].plot()
df['종가'].rolling(7).mean().plot(figsize=(15,7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1244ed090>




![png](/assets/fig/2019-11-11/output_23_1.png)


이동 평균에는 단순 이동평균, 선형 이동평균, 지수 이동평균 등이 있는데, 이들의 특징은 다음과 같다.  

1. 단순 이동 평균 : 특정 기간의 평균을 단순하게 구하는 것
2. 선형 이동 평균 : 특정 기간에 가중치를 곱해서 평균을 구하는 것
3. 지수 이동 평균 : 선형 이동 평균과 비슷하지만, 모든 기간에 가중치가 가해진다는 점에 차이가 있다.

지수 이동 평균은 새로운 값에 민감하다는 점에서 단순 이동 평균보다 유용하게 쓰일 수 있다.


```python
# 별도 컬럼으로 추가해서 범례를 확인해보자.
df['종가: 지수 이동 평균'] = df['종가'].ewm(7).mean()
df[['종가','종가: 지수 이동 평균']].plot(figsize=(15,7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123a76710>




![png](/assets/fig/2019-11-11/output_25_1.png)


### d. Visualizing  

예쁜 그래프를 그리는 것 역시 시각화의 중요한 요소이다.


```python
df[['거래량','종가']].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1244bc6d0>




![png](/assets/fig/2019-11-11/output_27_1.png)


이를테면 거래량과 종가는 단위가 크게 다르기 때문에, 같은 plot에 그리면 위와 같이 알아보기 어려운 문제가 있다. 따라서, 새로운 축을 하나 더 설정해줄 필요가 있다.


```python
df[['거래량','종가']].plot(secondary_y=['거래량'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122f52a90>




![png](/assets/fig/2019-11-11/output_29_1.png)


다음과 같이 크기를 조절할 수도 있고, 라벨링을 할 수도 있다.  
Grid를 넣어 보기 쉽게 할 수도 있다.


```python
fig, ax = plt.subplots()
ax.plot_date(df.index, df['종가'],'-')
ax.yaxis.grid(True)
ax.xaxis.grid(True)
fig.autofmt_xdate() # Auto fixes the overlap!
plt.tight_layout()
plt.ylabel('종가')
plt.xlabel('날짜')
plt.title('네이버')
plt.show()
```


![png](/assets/fig/2019-11-11/output_31_0.png)


## 3. 주식 예측  

본격적으로 주식을 예측해보자.

### a. Simple Moving Average  

Moving Average 구하는 방법에 대해서는 2-c의 rolling mean에서 알아보았다.


```python
df['6M SMA'] = df['종가'].rolling(window=6).mean()
df['12M SMA'] = df['종가'].rolling(window=12).mean()
```


```python
df[['종가', '6M SMA', '12M SMA']].plot(figsize=(12,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123004e90>




![png](/assets/fig/2019-11-11/output_35_1.png)


### b. Weighted Moving Average


```python
df['6M WMA'] = df['종가'].ewm(span=6).mean()
df['12M WMA'] = df['종가'].ewm(span=12).mean()
```


```python
df[['종가', '6M WMA', '12M WMA']].plot(figsize=(12,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122f1e810>




![png](/assets/fig/2019-11-11/output_38_1.png)


### c. Simple Exponential Smoothing

지수 평활을 사용하여 얻은 예측값은 과거 관측값의 가중평균이다. 이 중에서 단순 지수평활은 추세나 계절성 패턴이 없는 데이터를 예측할 때 쓰기 좋다. (3-b과 사실상 동일하다)


```python
from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np
train = df[:'2019-09']
test = df['2019-10':]
```


```python
train['종가'].plot(figsize=(12,8))
test['종가'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12169dd50>




![png](/assets/fig/2019-11-11/output_41_1.png)



```python
ses_model = SimpleExpSmoothing(np.asarray(train['종가']))
```


```python
ses_result = ses_model.fit()
```


```python
y_hat = test.copy()
```


```python
y_hat['SES'] = ses_result.forecast(len(test))
```


```python
plt.figure(figsize=(12,8))
plt.plot(train['종가'], label='Train')
plt.plot(test['종가'], label='Test')
plt.plot(y_hat['SES'], label='Simple Exp Smoothing')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1215cb290>




![png](/assets/fig/2019-11-11/output_46_1.png)



```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test['종가'], y_hat['SES']))
```


```python
print("평균 제곱근 편차:", rmse)
```

    평균 제곱근 편차: 4991.063442365193


### d. ARIMA

ARIMA는 Autoregressive Integrated Moving Average의 약자로, 자기 회귀 누적 이동 평균 쯤으로 해석할 수 있다. 자기 회귀와 이동 평균을 모두 고려하는 모델이라고 보면 된다.  
AR : 이전 관측값의 오차항이 이후 관측값에 영향을 주는 모형  
MA : 이동 평균 모형  
ARIMA(p, d, q) 값은 어떤 값을 선택하냐에 따라 모델의 성능이 달라진다.


```python
import statsmodels.api as sm
```


```python
arima = sm.tsa.statespace.SARIMAX(train['종가'],
                                  order=(1,0,2),
                                  seasonal_order=(0,1,0,12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
```

    /usr/local/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:219: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)



```python
arima_result = arima.fit()
```

주의해야할 점은, ARIMA는 시간의 빈도가 일정해야한다.
주 5일밖에 열리지 않는 (그마저도 빨간 날은 쉬는) 주식 장을 예측할 때는 난감한 일이 아닐 수가 없다. 따라서, 값을 적절하게 바꿔줘야 할 필요가 있다.


```python
ARIMA = arima_result.predict(start=len(train), end=len(train)+len(test)-1, dynamic=True)
```

    /usr/local/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:576: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      ValueWarning)



```python
ARIMA = pd.DataFrame(ARIMA, columns = ['ARIMA']).set_index(y_hat.index)
```


```python
y_hat = pd.merge(y_hat, ARIMA, left_index=True, right_index=True)
```


```python
plt.figure(figsize=(12,8))
plt.plot(train['종가'], label='Train')
plt.plot(test['종가'], label='Test')
plt.plot(y_hat['ARIMA'], label='ARIMA')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1208da090>




![png](/assets/fig/2019-11-11/output_57_1.png)



```python
rmse = np.sqrt(mean_squared_error(test['종가'], y_hat['ARIMA']))
```


```python
print("평균 제곱근 편차:", rmse)
```

    평균 제곱근 편차: 6184.27030528862



그래서 내일 오르냐고? 아직 모르겠다! :kissing: