import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import datetime

# Function to scrape company names, sector information from Wikipedia
def get_world_company_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable"})
    companies = []
    for row in table.findAll("tr")[1:]:
        symbol = row.findAll("td")[0].text.strip()
        company = row.findAll("td")[1].text.strip()
        sector = row.findAll("td")[3].text.strip()
        companies.append((symbol, company, sector))
    return pd.DataFrame(companies, columns=['Symbol', 'Company', 'Sector'])


# Load the pre-trained model
model = load_model('Stock Predictions Model.keras')

# Get world company information
world_company_info = get_world_company_info()


# Function to get sector of a symbol
def get_sector(symbol):
    sector = world_company_info[world_company_info['Symbol'] == symbol]['Sector'].values
    if len(sector) > 0:
        return sector[0]
    else:
        return "Unknown"


# Function to get stock data and sector
def get_stock_data(symbol, start, end):
    data = yf.download(symbol, start, end)
    sector = get_sector(symbol)
    return data, sector


# Function to calculate moving averages
def calculate_moving_averages(data):
    ma_50_days = data['Close'].rolling(50).mean()
    ma_100_days = data['Close'].rolling(100).mean()
    ma_200_days = data['Close'].rolling(200).mean()
    return ma_50_days, ma_100_days, ma_200_days


# Function to analyze stock data
def analyze_stock(data):
    # Calculate returns
    returns = data['Close'].pct_change()

    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility

    # Calculate average daily return
    avg_daily_return = returns.mean()

    # Calculate cumulative return
    cumulative_return = (data['Close'][-1] / data['Close'][0]) - 1

    return {
        'Volatility': volatility,
        'Average Daily Return': avg_daily_return,
        'Cumulative Return': cumulative_return
    }


# Streamlit layout customization
st.set_page_config(layout="wide")

# Header
st.title('SP Financial Analytics Platform')

# Add stock market background
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('digital-financial-chart-interface-big-data-analysis-platform-background_432516-5656.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title('Select Options')

# Search bar for company name
search_query = st.sidebar.text_input('Search for Company Name (* NYSE Only )')

# Filter companies based on search query
search_query = search_query.strip().rstrip('*')  # Remove trailing '*'
filtered_companies = world_company_info[world_company_info['Company'].str.contains(search_query, case=False)]

# Add the asterisk sign to the search query display
if search_query.endswith('*'):
    st.sidebar.write("Searching only for tickers (ASE)")


# Dropdown for selecting sector
selected_sector = st.sidebar.selectbox('Select Sector', filtered_companies['Sector'].unique())

# Get companies in the selected sector
sector_companies = filtered_companies[filtered_companies['Sector'] == selected_sector]['Company'].tolist()

# Dropdown for selecting company
selected_company = st.sidebar.selectbox('Select Company', sorted(set(sector_companies)))

# Get selected company symbol and sector
selected_company_info = filtered_companies[filtered_companies['Company'] == selected_company].squeeze()

# Date range for data retrieval
start = st.sidebar.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
end = datetime.date.today()  # Set end date to current date

# Display latest stock data for selected company
st.sidebar.subheader('Latest Stock Data')
latest_data, _ = get_stock_data(selected_company_info['Symbol'], end - pd.DateOffset(days=1), end)
st.sidebar.write(selected_company_info['Company'])
st.sidebar.write(latest_data.tail(1))

# Download stock data
data, sector = get_stock_data(selected_company_info['Symbol'], start, end)

# Display stock data
st.header('Stock Data for ' + selected_company_info['Company'])
st.write(data)

# Display sector
st.sidebar.write(f"**Sector:** {selected_company_info['Sector']}")

# Calculate moving averages
ma_50_days, ma_100_days, ma_200_days = calculate_moving_averages(data)

# Plot original price with moving averages
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Original Price', color='blue')
ax.plot(ma_50_days.index, ma_50_days, label='MA 50 Days', color='red')
ax.plot(ma_100_days.index, ma_100_days, label='MA 100 Days', color='green')
ax.plot(ma_200_days.index, ma_200_days, label='MA 200 Days', color='orange')
ax.set_title('Original Price with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences
X = []
y = []
n_past = 100

for i in range(n_past, len(data_scaled)):
    X.append(data_scaled[i - n_past:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Make predictions
try:
    predicted = model.predict(X)
    predicted = scaler.inverse_transform(predicted)

    # Plot predicted price for 2024
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index[n_past:], data['Close'].values[n_past:], label='Original Price', color='blue')
    ax.plot(data.index[n_past:], predicted, label='Predicted Price', color='red')
    ax.set_title('Predicted Price for 2024-2025')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred while making predictions: {str(e)}")

# Show selected company info
st.sidebar.header('Selected Company Info')
st.sidebar.write(selected_company_info)
