#  Calendar Anomalies Study

This project is a Python-based statistical analysis tool that tests famous stock market myths using hypothesis testing and historical data.

## About This Project
In quantitative finance, "Calendar Anomalies" are recurring patterns that seem to contradict the Efficient Market Hypothesis. Two of the most famous examples are the "Weekend Effect" (the theory that stocks tend to drop on Mondays) and the "January Effect" (the theory that stocks surge in January due to tax-loss harvesting). Instead of relying on anecdotes, this tool pulls over 20 years of S&P 500 data and uses T-Tests to determine if these patterns are statistically significant or just random noise.

##  Features
* **Automated Data Retrieval:** Fetches 24 years of daily S&P 500 data (2000-2024) via Yahoo Finance to ensure a large sample size.

* **Statistical Rigor:** Uses scipy.stats to perform independent 2-sample T-Tests, generating P-values to validate results.

* **Feature Engineering:** Extracts temporal features (Day Name, Month Name) from raw time-series data.

* **Visual Analysis:** Generates bar charts comparing average returns by Day of Week and Month, annotated with statistical findings.

##  Tech Stack
* **Data Acquisition:** `yfinance` (Historical market data)
* **Data Processing:** `pandas` (Time-series manipulation, DateTime extraction)
* **Statistics:** `scipy` (T-Tests, P-Value calculation)
* **Visualization:** `matplotlib`, `seaborn` (Statistical plotting)

##  How to Use
# 1. Prerequisites
* Python 3.8+
* Git


# 2. Setup & Installation
Clone this repository and set up the virtual environment:

```bash
# 1. Clone the repository to your local machine
git clone https://github.com/ClaysonV/Calendar-Effects-Study.git
cd Calendar-Effects-Study

# 2. Create a Python virtual environment
python -m venv venv

# 3. Activate the environment
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 4. Install the required libraries
pip install -r requirements.txt

```

# 3. Running the Study
You can run the script immediately. It defaults to analyzing SPY (S&P 500), but you can modify the ticker variable in the script if you want to test other assets (like BTC-USD).
```bash
Python

# --- Part 1: Data Acquisition ---
ticker = "SPY" 
# Change to "QQQ", "GLD", or "BTC-USD" to test other markets
```
Once ready, simply run the script from your terminal:
```bash
python calendar_effects.py
```

##  Core Concepts Applied
* **Hypothesis Testing:** Formulating Null vs. Alternative hypotheses and using T-statistics to reject or accept them.
* **Statistical Significance:** Interpreting P-values (< 0.05) to distinguish between signal and noise.
* **Behavioral Finance:** analyzing market inefficiencies driven by human behavior and institutional cycles.
* **Data Engineering:** Converting raw timestamps into categorical features (Monday, Tuesday, January, etc.).

##  Screenshots
* <img width="1920" height="1015" alt="Calendar Study Preview" src="calendar Effects study.png" />
