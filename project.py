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

