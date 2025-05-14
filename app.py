# Importing necessary libraries
import streamlit as st  # For creating interactive web app
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import requests  # To fetch data from web APIs
from prophet import Prophet  # For time series forecasting
from prophet.plot import plot_plotly  # For interactive Prophet plots
import plotly.express as px  # For bar, pie, and treemap charts
import plotly.graph_objects as go  # For more customizable charts
from sklearn.linear_model import LinearRegression  # Not used here, but generally for regression
from scipy.optimize import minimize  # For optimizing portfolio weights
from fpdf import FPDF  # For generating PDF reports
import matplotlib.pyplot as plt  # For bar charts
import seaborn as sns  # For heatmaps and other statistical plots
import tempfile  # To create a temporary file for PDF
import os  # File path operations
import time  # Sleep/pause

st.set_page_config(layout="wide", page_title="Crypto Portfolio Optimization", page_icon="📈")


# Function: Fetch Historical Data (CoinGecko)

def get_historical_data(coin_id, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error(f"Failed to fetch data for {coin_id}. Error: {response.status_code}")
        return pd.Series(dtype=float)

    data = response.json()

    if 'prices' not in data or not data['prices']:
        st.warning(f"No price data found for {coin_id}.")
        return pd.Series(dtype=float)

    prices = [price[1] for price in data['prices']]
    dates = [pd.to_datetime(price[0], unit='ms') for price in data['prices']]
    return pd.Series(prices, index=dates, name=coin_id)


# Function: Process Returns & Stats

def process_data(df):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return returns, mean_returns, cov_matrix


# Function: Portfolio Optimization (MPT)

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -1 * ((port_return - 0.02) / port_vol)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@st.cache_data
def get_all_coin_ids():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    data = response.json()
    # Return dictionary of coin name and id
    return {coin['name'].title(): coin['id'] for coin in data}


# Function: Fetch Trending Coins

def get_trending_coins():
    url = "https://api.coingecko.com/api/v3/search/trending"
    response = requests.get(url)
    data = response.json()
    trending_coins = [coin['item']['name'] for coin in data['coins']]
    return trending_coins


# Function: Forecast with Prophet
def run_prophet(coin_series):
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': coin_series.index,
        'y': coin_series.values
    })

    # Initialize and train Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dates and forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Create custom Plotly figure
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual',
                             line=dict(color='khaki', width=2)))

    # Forecast line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast',
                             line=dict(color='hotpink', dash='dash')))

    # Confidence interval (shaded area between yhat_upper and yhat_lower)
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0,255,255,0.2)',  # Cyan with transparency
        line=dict(color='lightcoral', width=0),
        name='Confidence Interval',
        showlegend=True
    ))

    # Customize layout
    fig.update_layout(
        title="📈 Prophet Forecast (30 Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        width=700,
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )

    return model, forecast, fig



# Function: Investment Suggestion

def get_trend(forecast):
    if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-30]:
        return "up"
    return "down"

def suggest_investment(expected_return, sharpe_ratio, trend):
    if expected_return > 0.10 and sharpe_ratio > 1 and trend == "up":
        return "✅ Suggestion: Good time to invest!"
    else:
        return "⚠️ Suggestion: Hold or avoid for now."
    


# Function to display portfolio allocation as a pie chart

def show_allocation_charts(allocation, investment_amount):
    allocation_df = pd.DataFrame({
        'Coin': list(allocation.keys()),
        'Weight': list(allocation.values()),
    })
    allocation_df['Investment ($)'] = allocation_df['Weight'] * investment_amount

    st.subheader("💼 Portfolio Allocation - Choose View")
    view = st.radio("Select View", ["Bar Chart", "Donut Chart", "Treemap", "Interactive Pie Chart"], horizontal=True)

    if view == "Bar Chart":
        fig = px.bar(allocation_df, x="Coin", y="Investment ($)", color="Coin",
                     text="Investment ($)", title="Investment Distribution (Bar Chart)")
        st.plotly_chart(fig)

    elif view == "Donut Chart":
        fig = go.Figure(data=[go.Pie(labels=allocation_df['Coin'],
                                     values=allocation_df['Weight'],
                                     hole=.4,
                                     textinfo='label+percent',
                                     title="Donut Chart - Allocation Weights")])
        st.plotly_chart(fig)

    elif view == "Treemap":
        fig = px.treemap(allocation_df, path=['Coin'], values='Investment ($)',
                         color='Investment ($)', title="Treemap - Investment Breakdown")
        st.plotly_chart(fig)

    elif view == "Interactive Pie Chart":
        fig = px.pie(allocation_df, names='Coin', values='Weight',
                     title='Portfolio Allocation - Pie Chart',
                     hover_data=['Investment ($)'], hole=0)
        st.plotly_chart(fig)



# Function to display the metrics (Expected Return, Volatility, Sharpe Ratio) as a bar chart

def plot_performance_metrics(expected_return, volatility, sharpe):
    metrics = ["Expected Return", "Volatility", "Sharpe Ratio"]
    values = [expected_return, volatility, sharpe]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(metrics, values, color=['#4CAF50', '#FF6347', '#FFD700'], edgecolor="black")
    ax.set_ylabel('Percentage/Ratio', fontsize=7)
    ax.set_title('Portfolio Performance Metrics', fontsize=16, fontweight='bold', color="#333333")
    ax.set_ylim(0, max(values) + 0.1)  # Set y-axis limit to leave some space at the top
    ax.set_xlabel('Metrics', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)



# Function: Generate PDF Report

def generate_pdf(portfolio_data, suggestion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Crypto Portfolio Optimization Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt="Portfolio Allocation:", ln=True)
    for k, v in portfolio_data.items():
        pdf.cell(200, 10, txt=f"{k}: {v:.2%}", ln=True)

    pdf.ln(10)

    # Remove emojis and convert suggestion to ASCII
    if "✅" in suggestion or "⚠️" in suggestion:
        suggestion_text = suggestion.split(":")[-1].strip()
    else:
        suggestion_text = suggestion

    pdf.cell(200, 10, txt=f"Investment Suggestion: {suggestion_text}", ln=True)

    # Use temporary file path
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "crypto_report.pdf")
    pdf.output(path)
    return path


# Streamlit App Interface

st.title("📈 Crypto Portfolio Optimization with Forecasting")

# Get trending coins
trending_coins = get_trending_coins()
st.subheader("🔥 Trending Coins")
st.write(", ".join(trending_coins))

# Investment Amount Input
investment_amount = st.number_input("Enter Investment Amount (USD)", min_value=1, step=1, value=1000, key="investment_amount")

st.subheader("🪙 Choose Cryptocurrencies for Portfolio")
coin_dict = get_all_coin_ids()
coin_names = list(coin_dict.keys())

selected_names = st.multiselect("Select Coins", coin_names, default=["Bitcoin", "Ethereum"])
coins = [coin_dict[name] for name in selected_names]

# Run Optimization Button
if st.button("Run Optimization") and coins:
    price_df = pd.DataFrame()
    for coin in coins:
        series = get_historical_data(coin)
        if not series.empty:
            price_df[coin] = series
        time.sleep(1)

    if price_df.empty:
        st.error("No valid price data found. Please try different coins.")
        st.stop()

    returns, mean_returns, cov_matrix = process_data(price_df)
    weights = optimize_portfolio(mean_returns, cov_matrix)
    expected_return = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (expected_return - 0.02) / volatility
    allocation = dict(zip(coins, weights))

    # Store in session state
    st.session_state["price_df"] = price_df
    st.session_state["returns"] = returns
    st.session_state["mean_returns"] = mean_returns
    st.session_state["cov_matrix"] = cov_matrix
    st.session_state["weights"] = weights
    st.session_state["expected_return"] = expected_return
    st.session_state["volatility"] = volatility
    st.session_state["sharpe"] = sharpe
    st.session_state["allocation"] = allocation


# 🔁 Show result if already available (persistent after button click)
if "allocation" in st.session_state:
    st.subheader("📊 Portfolio Allocation")
    show_allocation_charts(st.session_state["allocation"], investment_amount)

    st.subheader("📈 Portfolio Performance Metrics")
    st.metric("Expected Return", f"{st.session_state['expected_return']:.2%}")
    st.metric("Volatility", f"{st.session_state['volatility']:.2%}")
    st.metric("Sharpe Ratio", f"{st.session_state['sharpe']:.2f}")

    plot_performance_metrics(st.session_state["expected_return"] * 100,
                             st.session_state["volatility"] * 100,
                             st.session_state["sharpe"])

    st.subheader("📈 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(st.session_state["returns"].corr(), annot=True, cmap="coolwarm", ax=ax1,
                fmt='.2f', linewidths=1, cbar_kws={'shrink': 0.8})
    ax1.set_title("Correlation Heatmap", fontsize=10, fontweight='normal', color="#333333")
    st.pyplot(fig1)

    st.subheader("📉 Forecast Future Price")
    selected_coin = st.selectbox("Choose a coin for forecasting", st.session_state["price_df"].columns.tolist())
    model, forecast, fig2 = run_prophet(st.session_state["price_df"][selected_coin])
    st.plotly_chart(fig2)

    trend = get_trend(forecast)
    suggestion = suggest_investment(st.session_state["expected_return"], st.session_state["sharpe"], trend)
    st.subheader("📢 Investment Suggestion")
    st.success(suggestion)

    pdf_path = generate_pdf(st.session_state["allocation"], suggestion)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="Crypto_Report.pdf")

else:
    st.info("Enter coins and click Run Optimization.")