import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from chatbot import chatbot
import pybase64
from keras.initializers import Orthogonal
import datetime





image_path = 'chat_icon.png'
with open(image_path, "rb") as image_file:
    # Using pybase64 for encoding
    encoded_string = pybase64.b64encode(image_file.read()).decode()

st.markdown("""
<style>
/* Targeting buttons that act as tabs */
button[data-baseweb="tab"] {
    color: #00FFFF !important; /* Example: changing the tab text color */
}


        
</style>
""", unsafe_allow_html=True)

# for chatbot icon
st.markdown(f"""
<style>
button[data-testid="baseButton-headerNoPadding"] {{
    visibility: hidden !important;
}}

button[data-testid="baseButton-headerNoPadding"]::after {{
    content: url(data:image/png;base64,{encoded_string});
    visibility: visible !important;
    display: block !important;
    margin:20px !important;
    
}}
</style>
""", unsafe_allow_html=True)
# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return pd.DataFrame(data)

# Function to prepare data for LSTM model
def prepare_data(df, feature='Close', time_steps=100):
    data = df.loc[:, feature].values
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    x, y = [], []
    for i in range(time_steps, len(data_scaled)):
        x.append(data_scaled[i-time_steps:i, 0])
        y.append(data_scaled[i, 0])

    return np.array(x), np.array(y), scaler



def plot_predictions(y_test, y_predicted, dates, home, method='Matplotlib', ohlc_data=None):
    if method == 'Matplotlib':
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_test, 'b', label="Original Price")
        plt.plot(dates, y_predicted, 'r', label="Predicted Price")
        plt.title("Stock Price Prediction")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        home.pyplot(plt)
    elif method == 'Plotly':
        chart_type = st.selectbox("Select Chart Type", ['line', 'candlestick'])
        if chart_type == 'line':
            trace1 = go.Scatter(x=dates, y=y_test, mode='lines', name='Original Price', line=dict(color='blue'))
            trace2 = go.Scatter(x=dates, y=y_predicted, mode='lines', name='Predicted Price', line=dict(color='red'))
            fig = go.Figure(data=[trace1, trace2])
        elif chart_type == 'candlestick' and ohlc_data is not None:
            fig = go.Figure(data=[go.Candlestick(x=dates,
                                                open=ohlc_data['open'], high=ohlc_data['high'],
                                                low=ohlc_data['low'], close=ohlc_data['close'],
                                                 name='Market Data')])
            fig.add_trace(go.Scatter(x=dates, y=y_predicted, mode='lines', name='Predicted Price', line=dict(color='red')))
        fig.update_layout(title="Stock Price Prediction", xaxis_title='Time', yaxis_title='Price')
        home.plotly_chart(fig)


def main_page(about_content):
    st.title("Tuber Market")
    home , about = st.tabs(['Home','About Us'])
    
    
    with about:
        st.write(about_content)
    with home:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL")

        start_date = "2015-01-01"
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        df_stock = download_stock_data(ticker, start_date, end_date)
        ohlc_data = {
            'open': df_stock['Open'],
            'high': df_stock['High'],
            'low': df_stock['Low'],
            'close': df_stock['Close']
        }


        st.write(df_stock.tail())

        time_steps = 100
        x_test, y_test, scaler = prepare_data(df_stock, 'Close', time_steps=time_steps)

    
        model = load_model('prediction.h5')
   
        y_predicted = model.predict(x_test)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_predicted = scaler.inverse_transform(y_predicted).flatten()

        dates = df_stock.index[-len(y_test):]  
        # Plot predictions
        plot_method = st.selectbox("Select Plot Method", ['Matplotlib', 'Plotly'])
        # plot_predictions(y_test, y_predicted, dates,home,method=plot_method)
        plot_predictions(y_test, y_predicted, dates, home, method=plot_method, ohlc_data=ohlc_data)


        data_filtered = df_stock[['Close']]
        data_filtered['Close'] = data_filtered['Close'].astype(float)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_filtered)

        # Create the scaled training data set
        train_data = scaled_data

        # Split the data into x_train and y_train data sets
        time_step = 100
        X_train = []
        y_train = []
        for i in range(time_step, len(train_data)):
            X_train.append(train_data[i-time_step:i, 0])
            y_train.append(train_data[i, 0])

        # Convert the x_train and y_train to numpy arrays 
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape the data into the shape accepted by the LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


        model = load_model('prediction.h5')
        # Create the test data set
        # Create a new array containing scaled values from the end of the training data set
        test_data = scaled_data[-time_step:, :]

        # Create the data sets x_test and y_test
        x_test = []
        x_test.append(test_data[:, 0])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the predicted scaled price
        predicted_price = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        st.write(f"Predicted stock price for the next day after {end_date}: {predicted_price[0][0]}")


about_content = """# About Us \n Welcome to our innovative project, the "Stock Market Screener" – a tool designed to empower investors and traders with the ability to filter and select stocks based on specific criteria, helping them make informed investment decisions. This project is a culmination of our dedication and hard work in pursuit of our Diploma in Computer Engineering, presented to the prestigious Maharashtra State Board of Technical Education, Mumbai.

## Our Journey\n
The journey of developing the "Stock Market Screener" began in the academic year 2023-2024, under the roof of the Government Polytechnic Arvi Dist.-Wardha, a renowned institution known for fostering innovation and technical excellence. This project is not just a requirement for our diploma; it's a reflection of our passion for combining the fields of computer engineering and finance to create tools that can significantly impact the way financial markets are analyzed.

## The Team\n
Our team is a trio of dedicated and enthusiastic computer engineering students, each bringing a unique set of skills and perspectives to the project:

##### Rushikesh Burankar (2101320111)

##### Tejas Tikkas (2101320097)

##### Sanket Nasare (2101320100)

## Guided by Excellence\n
The project was brought to fruition under the expert guidance of 
#####    Mr. S.U. Rathod Sir
a distinguished lecturer in the Department of Computer Engineering. His mentorship has been invaluable, providing us with the insights and encouragement needed to navigate the challenges of developing a sophisticated financial tool.

## Our Vision\n
The "Stock Market Screener" is more than just a project for us; it's a step towards democratizing financial information and tools, making them accessible to everyone. We believe in the power of technology to transform lives, and through this project, we aim to contribute to the financial well-being of individuals around the globe.

## Connect with Us \n
We are always looking to connect with fellow tech enthusiasts and financial experts. Whether you're interested in learning more about our project, sharing feedback, or discussing potential collaborations, feel free to reach out.

Thank you for visiting our website. We are excited to embark on this journey with you, exploring the vast possibilities of the stock market with the help of our Stock Market Screener.

"""


if __name__ == '__main__':
    
    chatbot()
    main_page(about_content)
   