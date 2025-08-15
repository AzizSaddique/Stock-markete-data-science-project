# import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# tilte
app_title = "Stock Price Analysis App"
st.title(app_title)

st.subheader("Analyze stock prices and visualize trends")
# sidebar

st.image('https://images.timesnownews.com/thumb/msid-151515654,thumbsize-36610,width-1280,height-720,resizemode-75/151515654.jpg')
st.sidebar.header("User Input")
start_date=st.sidebar.date_input('Start date', value=dt.date(2020, 1, 1))
end_date=st.sidebar.date_input('End date', value=dt.date(2020, 12, 31))

# add ticker in symbol
ticker_list=("AAPL , MSFT, GOOGL, AMZN, TSLA, FB, NFLX, NVDA, BRK-A, V, JNJ, WMT, PG, UNH, DIS, MA, HD, PYPL, VZ, T, KO")
ticker=st.sidebar.selectbox('Select company...',ticker_list.split(', '))
# fetch data from user input using yfinance library
data=yf.download(ticker,start=start_date,end=end_date)


# add date as an column in dataframe
data.insert(0, 'Date', data.index)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date,'to',end_date)
st.write(data)

# plot the data
st.header('Data visualization')
st.subheader('Plot the data')
# Ensure the columns are properly formatted
data.columns = [
    ' '.join(col).strip() if isinstance(col, tuple) else col
    for col in data.columns
]

# Now your columns will look like:
# ['Date', 'Adj Close AAPL', 'Close AAPL', 'High AAPL', 'Low AAPL', 'Open AAPL', 'Volume AAPL']

# Use all columns except Date for y-axis
fig = px.line(data,x='Date', y=data.columns[1:],title=f'{ticker} Stock Price'
)
st.plotly_chart(fig)

# add a selectbox to select the column from data
column=st.selectbox('Select column to plot', data.columns[1:])

# submiting the data
data=data[['Date', column]]
st.write('Selected the data')
st.write(data)

# adf check test stationarity
st.subheader('ADF Test for Stationariy')

st.write(adfuller(data[column])[1] <0.05)

# Lets decompose the data
st.subheader('Decompose the data')
decomposition=seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())

# make same plot in plotly
st.write('## ploting the decomposition in plotly')
st.plotly_chart(px.line (x=data['Date'], y=decomposition.trend, title='Trend',width=1200,height=400,labels={'x': 'Date', 'y': 'price'}).update_traces(line_color='blue'))
st.plotly_chart(px.line (x=data['Date'], y=decomposition.trend, title='Seasolality',width=1200,height=400,labels={'x': 'Date', 'y': 'price'}).update_traces(line_color='green'))
st.plotly_chart(px.line (x=data['Date'], y=decomposition.trend, title='Residuals',width=1200,height=400,labels={'x': 'Date', 'y': 'price'}).update_traces(line_color='red',line_dash='dot'))

# lets run the model
# user input for three parameters for the model and seasonal order
p=st.slider('Select p', min_value=0, max_value=5, value=1)
d=st.slider('Select d', min_value=0, max_value=5, value=1)
q=st.slider('Select q', min_value=0, max_value=5, value=2)
seasonal_order=st.number_input('Select seasonal order', min_value=0, max_value=24, value=12)

model=sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model=model.fit()
# print the summary of the model
st.header('Model summery')
st.write(model.summary())
st.write('-----')

st.write(" <p style='color:green; font-size:50px; fornt-weight:bold;'>Forcasting the data</p> ",unsafe_allow_html=True)
# predict the future values (forcosting)
forcast_period=st.number_input('## Enter forcast period in days',value=10)

# predict the future values
predictions=model.get_prediction(start=len(data), end=len(data) + forcast_period)
predictions=predictions.predicted_mean

# add index to results dataframe as date
predictions.index=pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write('## Predictions',predictions)
st.write('## Actual Data',data)

#  lets plot the data
fig=go.Figure()
# add actual data to the plot
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual Data', line=dict(color='blue')))
#  add predict the data to the plot



fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Predicted Data', line=dict(color='red')))
# set the title and labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1400, height=600)
# Display the plot
st.plotly_chart(fig)
