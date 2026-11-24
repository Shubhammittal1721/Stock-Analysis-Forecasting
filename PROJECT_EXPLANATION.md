# Project Explanation & Comprehensive Viva Preparation Guide

**Project Title:** Stock Market Analysis and Time Series Forecasting Using Python  
**Target Stock:** Apple Inc. (`AAPL`)  
**Target Level:** B.Tech Computer Science & Engineering (3rd Year)  
**Academic Institution:** Bennett University  
**Subject:** Data Science in Python  

---

## Table of Contents
1. [Core Conceptual Explanations (Sections 1 - 26)](#part-1-core-project-concepts-and-architecture)
2. [Viva Questions & Answers (35+ Curated Questions)](#part-2-viva-questions-and-answers)
3. [5-Minute Project Speech for Professor / Viva Examiner](#part-3-5-minute-verbal-project-explanation)

---

# Part 1: Core Project Concepts and Architecture

### 1. Project Title
**"Stock Market Analysis and Time Series Forecasting Using Python"**

### 2. Project Objective
Collect 5 continuous years of daily financial stock market data, perform exploratory time-series data analysis (EDA), analyze price momentum and volatility, test for mathematical stationarity, engineer temporal features, build an **ARIMA(1, 1, 1)** forecasting model, benchmark it against a machine learning model (Random Forest), and evaluate forecast performance using industry-standard error metrics (MAE, RMSE, MAPE).

### 3. Problem Statement
Stock prices fluctuate continuously due to market sentiment, company performance, macroeconomic factors, and investor psychology. Predicting stock prices is challenging because financial market data exhibits non-stationarity, high noise, and volatility clustering. The academic problem is to determine whether historical sequential price patterns contain sufficient autoregressive signal to generate reliable short-term forecasts, and to understand the mathematical boundaries of time-series modeling.

### 4. Why This Project is a Time-Series Project
In standard tabular machine learning (like predicting house prices or customer churn), each record/row is assumed to be **independent and identically distributed (i.i.d.)**.  
In stock market data, rows are **not independent**—they are arranged chronologically:
- Today's price $P_t$ depends directly on yesterday's price $P_{t-1}$.
- Observations possess **temporal ordering** and **autocorrelation** (correlation with past values).
- Shuffling the rows destroys the underlying physical meaning of the data. Hence, this problem belongs strictly to the domain of **Time Series Analysis**.

### 5. Dataset Information
- **Asset:** Apple Inc. (`AAPL`), listed on the NASDAQ exchange.
- **Source:** Yahoo Finance API via the `yfinance` Python library.
- **Window:** 5 continuous historical years (October 2021 to October 2026, 1,255 daily sessions).
- **Features Captured:**
  1. `Date`: Calendar trading date (used as DateTimeIndex).
  2. `Open`: Opening price when trading commenced at 9:30 AM EST.
  3. `High`: Highest transaction price recorded during the trading day.
  4. `Low`: Lowest transaction price recorded during the trading day.
  5. `Close`: Final standard transaction price at market close (4:00 PM EST).
  6. `Adj Close`: Adjusted Close, factoring in corporate actions like dividend payouts and stock splits.
  7. `Volume`: Total number of shares transacted throughout the session.

### 6. All Libraries Used and Why
| Library | Purpose in this Project |
| :--- | :--- |
| **`pandas`** | Time-series data manipulation, Datetime indexing, rolling window calculations, and tabular summaries. |
| **`numpy`** | Vectorized mathematical operations, log calculations, array transformations, and RMSE computation. |
| **`matplotlib.pyplot`** | Core plotting engine for rendering multi-panel graphs, price trends, and error diagnostics. |
| **`seaborn`** | Statistical data visualization, kernel density estimation (KDE), and distribution plots. |
| **`yfinance`** | Automated extraction of historical stock market OHLCV data directly from Yahoo Finance servers. |
| **`statsmodels`** | Implementation of econometric and statistical time-series algorithms: `seasonal_decompose`, `adfuller` (ADF test), and `ARIMA`. |
| **`scikit-learn`** | Machine learning comparison baseline (`RandomForestRegressor`) and regression metrics (`MAE`, `RMSE`, `MAPE`). |

### 7. Explanation of Major Steps
1. **Data Acquisition:** Downloaded 5 years of daily AAPL records and safely cached to `data/raw/AAPL_5y.csv`.
2. **Data Understanding & Cleaning:** Checked datatypes, removed duplicates, handled non-trading days by aligning with business day frequency (`asfreq('B')`), and applied forward fill (`ffill`).
3. **Exploratory Data Analysis (EDA):** Visualized long-term trends, trading volume surges, and daily price spreads.
4. **Moving Average Analysis:** Smoothed market noise using 20-day, 50-day, and 200-day rolling averages to analyze trend regimes.
5. **Returns and Volatility Study:** Computed daily percentage returns and 20-day rolling annualized volatility, analyzing distribution skewness and kurtosis.
6. **Time Series Decomposition:** Deconstructed Apple's stock series into Trend, Seasonal, and Residual components.
7. **Stationarity Testing:** Applied the Augmented Dickey-Fuller (ADF) test on both raw price and differenced price series.
8. **Feature Engineering:** Built lag features (Lag 1, 2, 5) and rolling features for machine learning comparison.
9. **Chronological Splitting:** Partitioned the data into 80% sequential training and 20% out-of-sample testing data (strictly no shuffling).
10. **ARIMA Modeling:** Configured and fitted an ARIMA(1, 1, 1) model.
11. **Walk-Forward Validation:** Generated 1-step ahead test predictions and analyzed residual errors.
12. **Benchmarking with Random Forest:** Trained a Random Forest Regressor on the same chronological split.
13. **Evaluation:** Compared models via MAE, RMSE, and MAPE.
14. **Future Forecast:** Projected a 14-business-day forecast with a 95% confidence interval.

### 8. Key Empirical Findings
- **Raw Price Non-Stationary:** ADF statistic = `0.2414`, $p$-value = `0.9745` (> 0.05). Fail to reject $H_0 \implies$ Non-Stationary ($I(1)$).
- **First Difference Stationary:** ADF statistic = `-12.4733`, $p$-value = `0.000000` (< 0.05). Reject $H_0 \implies$ Strictly Stationary ($I(0)$). Mathematically proves $d = 1$.
- **ARIMA(1, 1, 1) Performance:** MAE = **\$3.07**, RMSE = **\$4.38**, MAPE = **1.05%** on the out-of-sample test set.
- **Random Forest Baseline:** MAE = **\$26.38**, RMSE = **\$36.30**, MAPE = **8.33%**. Tree models cannot extrapolate upward trends beyond training ceilings without differencing.

---

# Part 2: Viva Questions and Answers

### Q1. What is a time series?
**Ans:** A sequence of observations recorded chronologically at regular, successive time intervals where sequential order matters.

### Q2. Why is stock market data considered time-series data?
**Ans:** Because stock prices follow chronological order and exhibit temporal dependency and autocorrelation—today's price depends directly on yesterday's price.

### Q3. What is the fundamental difference between time series and standard regression?
**Ans:** Standard regression assumes data is independent and identically distributed (i.i.d.). Time-series data violates this assumption due to temporal autocorrelation.

### Q4. What is stationarity?
**Ans:** A time series is stationary if its mean, variance, and autocovariance remain constant over time.

### Q5. Why must we test stationarity before fitting ARIMA?
**Ans:** Non-stationary data has a time-dependent mean or variance, leading to spurious statistical relationships and explosive forecast error variances.

### Q6. What is the Augmented Dickey-Fuller (ADF) test?
**Ans:** An econometric hypothesis test that checks whether a unit root is present in a time series:
- Null Hypothesis ($H_0$): Unit root exists $\implies$ Non-stationary.
- Alternative Hypothesis ($H_1$): No unit root $\implies$ Stationary.
- Decision Rule: If $p \le 0.05$, reject $H_0$ (stationary).

### Q7. Was raw Apple stock price stationary in your project?
**Ans:** No. Raw AAPL closing price had an ADF p-value of $\approx 0.975$ (> 0.05), proving non-stationarity ($I(1)$).

### Q8. How did you make the series stationary?
**Ans:** By first differencing ($d=1$), which computes price changes ($P_t - P_{t-1}$). The differenced series had a p-value $< 0.0001$ (strictly stationary).

### Q9. What does ARIMA(p, d, q) stand for?
**Ans:** AutoRegressive Integrated Moving Average:
- $p = 1$: AutoRegressive order (1 lagged price term).
- $d = 1$: Integrated differencing (1 differencing step to achieve stationarity).
- $q = 1$: Moving Average order (1 lagged error shock term).

### Q10. Why can't we randomly split time-series data using `train_test_split(shuffle=True)`?
**Ans:** Random shuffling causes lookahead bias (data leakage)—future prices leak into past training data, yielding falsely inflated accuracy that collapses in reality.

### Q11. What is MAE and RMSE?
**Ans:** MAE is the average absolute dollar error. RMSE squares errors before averaging, penalizing sharp outlier spikes more heavily.

### Q12. Why did ARIMA outperform Random Forest in your project?
**Ans:** Decision trees cannot extrapolate values outside the numerical range seen in the training data. When stock prices rose to new highs, Random Forest predicted flat lines, whereas ARIMA with $d=1$ differencing naturally projected upward momentum.

---

# Part 3: 5-Minute Verbal Project Explanation

"Good morning, Professor. Today I am presenting our Data Science in Python project titled **'Stock Market Analysis and Time Series Forecasting Using Python'** for **Apple Inc. (AAPL)**.

### 1. Problem & Dataset
Stock price movements are notoriously noisy and volatile. The goal of this project is to analyze 5 years of daily historical trading data, extract trend and volatility patterns, and evaluate whether statistical time-series models can reliably forecast price momentum.

We extracted 1,255 trading days of daily AAPL records using the Yahoo Finance API. The data includes Open, High, Low, Close, Adjusted Close, and Volume.

### 2. Preprocessing & Cleanliness
In time-series analysis, data hygiene is paramount. We confirmed zero null values and zero duplicates. We then aligned the data to an explicit business-day frequency (`asfreq('B')`) using forward fill to handle weekend and market holiday closures, satisfying statsmodels modeling requirements.

### 3. Exploratory Analysis & Moving Averages
We visualized the 5-year closing price trend, trading volume, and intraday spreads. 

We computed 20-day, 50-day, and 200-day Simple Moving Averages. We observed that the 200-day MA acted as consistent long-term structural support during market pullbacks. Maintaining the 50-day MA above the 200-day MA confirmed a sustained bullish Golden Cross regime.

Analyzing daily returns revealed **volatility clustering**—periods of turbulence group together. Furthermore, the return distribution displayed a kurtosis of 4.10 (> 3.0), mathematically confirming **leptokurtic fat tails** where extreme drawdowns occur more frequently than a Gaussian bell curve predicts.

### 4. Stationarity & Econometric Testing
Statistical models like ARIMA require **stationarity** (constant mean and variance). 

We conducted the **Augmented Dickey-Fuller (ADF) test**:
- The raw closing price produced a p-value of **0.9745**, failing to reject the null hypothesis. This proved that raw stock prices are non-stationary ($I(1)$ process).
- Applying first differencing ($d=1$) yielded a p-value of **0.000000** ($< 0.0001$), strictly rejecting the null hypothesis. This mathematically justified setting the differencing parameter $d = 1$ in our ARIMA model.

### 5. Forecasting Models & Evaluation
We split the dataset strictly chronologically into 80% training (~1,004 days) and 20% out-of-sample testing (~251 days) to prevent lookahead bias.

We implemented an **ARIMA(1, 1, 1)** model with walk-forward validation and benchmarked it against a Random Forest Regressor trained on autoregressive lag features:
- ARIMA achieved a **Mean Absolute Error (MAE) of \$3.07** and a **MAPE of 1.05%** on the test set.
- Random Forest achieved an MAE of \$26.38 and MAPE of 8.33%. Decision trees failed because they cannot extrapolate upward trends beyond the historical training ceiling without differencing.

### 6. Future Projection & Limitations
We projected a 14-business-day future forecast with a 95% confidence interval. The widening confidence band illustrates that forecasting uncertainty accumulates over longer horizons.

In conclusion, our project demonstrates that while long-term stock prices follow an unpredictable random walk influenced by exogenous news shocks, short-term momentum can be captured with high accuracy using mathematically sound econometric models like ARIMA(1, 1, 1).

Thank you, Professor. We are now happy to answer your questions."
