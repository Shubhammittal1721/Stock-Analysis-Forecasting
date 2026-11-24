# Stock Market Analysis and Time Series Forecasting

An academic Data Science project in Python analyzing historical stock market behavior, trend regimes, volatility clustering, and price forecasting for Apple Inc. (`AAPL`) using time series decomposition, econometric stationarity testing (ADF), ARIMA, and machine learning benchmarking.

---

## 📌 Project Overview
This project performs an end-to-end time-series analysis on 5 years of daily trading data for **Apple Inc. (`AAPL`)**. It covers data ingestion via the Yahoo Finance API, data cleaning, exploratory data analysis, moving average trend analysis, return distribution modeling, additive time-series decomposition, Augmented Dickey-Fuller (ADF) stationarity testing, feature engineering, chronological train-test splitting, ARIMA forecasting, Random Forest regression benchmarking, and out-of-sample future projections with confidence intervals.

---

## 🎯 Objectives
- Extract 5 years of daily historical OHLCV data using `yfinance` with fallback caching.
- Conduct exploratory data analysis on price trajectories, trading volume, and intraday spreads.
- Analyze market momentum and trend reversals via 20-day, 50-day, and 200-day Simple Moving Averages.
- Examine daily percentage returns, volatility clustering, and distribution fat tails (kurtosis).
- Decompose the series into Trend, Seasonal, and Residual components using `statsmodels`.
- Formally evaluate stationarity using the Augmented Dickey-Fuller (ADF) test.
- Engineer autoregressive lag features and rolling window statistical aggregations.
- Apply a strict chronological train-test split (80% train / 20% test) to prevent lookahead bias.
- Train and evaluate an **ARIMA(1, 1, 1)** forecasting model.
- Benchmark ARIMA against a Random Forest Regressor using standard metrics (MAE, RMSE, MAPE).
- Forecast future out-of-sample trading days with a 95% confidence interval.

---

## 🛠️ Technologies & Libraries
- **Language:** Python 3.10+
- **Environment:** Jupyter Notebook
- **Core Libraries:**
  - `pandas`: Time-series indexing, rolling computations, data cleaning.
  - `numpy`: Vectorized mathematical calculations.
  - `matplotlib` & `seaborn`: Multi-panel visualizations, KDE density plots, diagnostics.
  - `yfinance`: Yahoo Finance market data acquisition.
  - `statsmodels`: Seasonal decomposition, ADF unit-root test, ARIMA modeling.
  - `scikit-learn`: Random Forest Regressor, error metrics (MAE, RMSE, MAPE).

---

## 📊 Dataset
- **Asset:** Apple Inc. (Ticker: `AAPL`)
- **Period:** ~5 years of daily historical trading data (~1,255 trading days).
- **Attributes:**
  - `Date`: Trading day timestamp (DateTimeIndex).
  - `Open`: Opening price ($).
  - `High`: Intraday peak price ($).
  - `Low`: Intraday lowest price ($).
  - `Close`: Final standard closing price ($).
  - `Adj Close`: Price adjusted for dividends and splits ($).
  - `Volume`: Total shares traded.
- **Local Cache:** `data/raw/AAPL_5y.csv` (guarantees offline reproducibility).

---

## 🔬 Methodology

```
┌────────────────────────────────────────────────────────┐
│ 1. Data Collection (yfinance API + Local CSV fallback) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Preprocessing & Alignment (Business Days 'B' + ffill│
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Exploratory Analysis & Moving Averages (20/50/200)   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Returns, Volatility & Seasonal Decomposition         │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Stationarity Analysis (ADF Test on Level & Diff)    │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6. Chronological Split (80% Train / 20% Test)          │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 7. Forecasting & Benchmarking (ARIMA vs Random Forest)  │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 8. Future 14-Day Forecast with 95% Confidence Interval │
└────────────────────────────────────────────────────────┘
```

1. **Data Cleaning:** Reindexed to business days (`asfreq('B')`), forward filled market holiday closures.
2. **Trend Analysis:** 20, 50, and 200-day Simple Moving Averages to identify golden/death crosses and institutional support levels.
3. **Volatility & Distribution:** Computed annualized rolling standard deviation; analyzed kurtosis and skewness for fat-tailed behavior.
4. **Decomposition:** Additive decomposition (`period=21`) isolating trend, monthly seasonality, and residual noise.
5. **Stationarity Testing:**
   - Raw Closing Price: $p \approx 0.95$ (Fail to reject $H_0 \implies$ Non-Stationary, $I(1)$).
   - First Differenced Series: $p < 0.0001$ (Reject $H_0 \implies$ Strictly Stationary, $I(0)$).
6. **Chronological Splitting:** Earliest 80% used for model training; untouched 20% retained for out-of-sample testing to strictly eliminate lookahead bias.
7. **Forecasting Models:**
   - **ARIMA(1, 1, 1):** Fit on training set, walk-forward state updates across test set.
   - **Random Forest Regressor:** 100 estimators, max depth 8, trained on Lag 1, Lag 2, Lag 5, MA7, and MA20.

---

## 📈 Results & Evaluation

| Metric | ARIMA(1, 1, 1) | Random Forest Regressor |
| :--- | :---: | :---: |
| **MAE (Mean Absolute Error)** | **~$3.07** | ~$26.12 |
| **RMSE (Root Mean Squared Error)** | **~$4.38** | ~$36.02 |
| **MAPE (Mean Absolute Percentage Error)** | **~1.5%** | ~8.25% |

### Key Findings
- **ARIMA Superiority for 1-Step Momentum:** The 1-step ahead ARIMA model tracked next-day price momentum with an average error of only ~\$3.07 per share (~1.5% relative error).
- **ML Lag Challenge:** Standard tree-based models like Random Forest without differencing struggle to extrapolate when test set prices move outside the range of historical training values.
- **Fat-Tailed Risk:** Return kurtosis significantly exceeds 3.0, proving that classical Gaussian assumptions underestimate large market drawdown events.
- **Widening Future Bounds:** The 14-day future projection illustrates rapidly expanding confidence intervals, highlighting the limits of multi-step forecasting.

---

## 📂 Project Structure

```
timeSeries/
├── app.py                              # Interactive Streamlit Web Application
├── stock_analysis_time_series.ipynb    # Main executed Jupyter notebook (all 19 sections)
├── PPT1_Stock_Market_Project_Overview.pptx # 8-slide Bennett University presentation
├── PPT2_Models_Implemented_and_Results.pptx # 9-slide results & models presentation
├── PROJECT_EXPLANATION.md              # Detailed viva guide (topics + Q&As + speech)
├── README.md                           # Project documentation & overview
├── data/
│   ├── raw/
│   │   └── AAPL_5y.csv                 # Cached 5-year AAPL historical data
│   └── processed/
```

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.10+ is installed. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn statsmodels streamlit plotly
```

### 2. Launch the Interactive Website (Streamlit Web App)
To launch the full interactive web dashboard in your browser:
```bash
streamlit run app.py
```
This opens the web app automatically at **`http://localhost:8501`**.

### 3. Launch the Jupyter Notebook
To run the step-by-step academic notebook:
```bash
jupyter notebook stock_analysis_time_series.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells end-to-end.

---

## ⚠️ Limitations & Academic Disclaimers
1. **Unforeseen Real-World News:** Statistical models only analyze past price patterns and cannot foresee earnings surprises, executive changes, regulatory updates, or geopolitical conflicts.
2. **Efficient Market Hypothesis (EMH):** Stock prices behave largely as a stochastic random walk with drift; pure technical patterns cannot consistently generate excess market returns (alpha).
3. **Stationarity & Linear Assumptions:** ARIMA assumes linear relationships among differenced lags, whereas financial markets exhibit regime shifts and asymmetric shocks.
4. **Educational Use Only:** All projections and analyses are for academic coursework and do **not** constitute financial or investment advice.

---

## 🎓 Viva Preparation
A companion document [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) is provided in the repository. It includes:
- Thorough breakdown of all 26 core concepts.
- 35+ frequently asked viva questions with clear, memorizable student answers.
- A natural 5-minute verbal explanation script tailored for speaking directly to an examining professor.
