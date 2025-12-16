import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
# Ignore all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Part 1: Data Acquisition ---
ticker = "SPY"
print(f"Acquiring 20+ years of historical data for {ticker}...")

# Added 'multi_level_index=False' to fix the column naming bug in new yfinance versions
data = yf.download(ticker, start="2000-01-01", end="2024-01-01", progress=False, multi_level_index=False)['Close']

# Calculate daily percentage returns
returns = data.pct_change().dropna() * 100

# --- Part 2: Data Preprocessing ---
returns_df = pd.DataFrame(returns)


returns_df.columns = ['Close'] 

# Feature Engineering
returns_df['Day_Name'] = returns.index.day_name()
returns_df['Month_Name'] = returns.index.month_name()

# --- Part 3: Hypothesis 1 - The "Weekend Effect" ---
monday_returns = returns_df[returns_df['Day_Name'] == 'Monday']['Close']
friday_returns = returns_df[returns_df['Day_Name'] == 'Friday']['Close']

t_stat, p_value_days = stats.ttest_ind(friday_returns, monday_returns, equal_var=False)

# --- Part 4: Hypothesis 2 - The "January Effect" ---
january_returns = returns_df[returns_df['Month_Name'] == 'January']['Close']
rest_of_year_returns = returns_df[returns_df['Month_Name'] != 'January']['Close']

t_stat_jan, p_value_jan = stats.ttest_ind(january_returns, rest_of_year_returns, equal_var=False)

# --- Part 5: Visualization ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Chart 1: Average Return by Day of Week
order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
sns.barplot(x='Day_Name', y='Close', data=returns_df, order=order_days, ax=ax1, errorbar=None, palette="viridis")

ax1.set_title(f"Average Return by Day of Week (2000-2023)\nP-Value (Fri vs Mon): {p_value_days:.4f}", color='white')
ax1.set_xlabel("")
ax1.set_ylabel("Avg Return (%)")
ax1.axhline(0, color='white', linestyle='--', alpha=0.5)

# Chart 2: Average Return by Month
order_months = ["January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"]
sns.barplot(x='Month_Name', y='Close', data=returns_df, order=order_months, ax=ax2, errorbar=None, palette="magma")

ax2.set_title(f"Average Return by Month\nP-Value (Jan vs Rest): {p_value_jan:.4f}", color='white')
ax2.set_xlabel("")
ax2.set_ylabel("Avg Return (%)")
ax2.axhline(0, color='white', linestyle='--', alpha=0.5)

plt.xticks(rotation=45) 

plt.tight_layout()
plt.show()

# --- Part 6: Reporting Results ---
print("\nSTATISTICAL ANOMALY REPORT")
print("-" * 40)
print(f"Dataset Range: 2000-01-01 to 2024-01-01")
print("-" * 40)

# Report: Day-of-Week Analysis
print(f"\n1. THE WEEKEND EFFECT (Friday vs Monday)")
print(f"   Mean Monday Return: {monday_returns.mean():.4f}%")
print(f"   Mean Friday Return: {friday_returns.mean():.4f}%")
print(f"   P-Value: {p_value_days:.5f}")

if p_value_days < 0.05:
    print("   CONCLUSION: Statistically Significant. The difference in means is unlikely due to chance.")
else:
    print("   CONCLUSION: Not Significant. Fail to reject the null hypothesis.")

# Report: January Effect Analysis
print(f"\n2. THE JANUARY EFFECT")
print(f"   Mean January Return: {january_returns.mean():.4f}%")
print(f"   Mean Rest of Year:   {rest_of_year_returns.mean():.4f}%")
print(f"   P-Value: {p_value_jan:.5f}")

if p_value_jan < 0.05:
    print("   CONCLUSION: Statistically Significant. January returns differ from the yearly average.")
else:
    print("   CONCLUSION: Not Significant. No statistical evidence of a January effect.")
print("-" * 40)