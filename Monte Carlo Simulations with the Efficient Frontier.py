#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo Simulations with the Efficient Frontier
# 

# ## Summary of Efficient Frontier
# 
#  The Efficient fronter is a set of optimal portfolios that offer the highest expected return for a defined level of risk. It provides a great visualization on how to choose an optimal portfolio mathematically. *Risk is defined as the assests actual return differing from our expected return.*

# ## Summary
# 
# I will simulate weights on individual companies within a given portfolio to obtain an understanding on what return to risk is desired by the individual.
# 
# I picked 10 or so companies that are spread out in their corresponding Industries such that we have a relatively "low" correlation with each other.

# ## Companies
# #### Google | NVIDIA | Facebook
# #### Wells Fargo | Pfizer | COKE
# #### Disney | IMAX | Catepillar
# #### Southwest Airlines

# In[1]:


import re
from io import StringIO
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np


# In[2]:


import yfinance as yf
import pandas as pd
from datetime import datetime

def get_historical_data(tickers):
    """
    This function returns a pd dataframe with all of the adjusted closing information
    """
    data = pd.DataFrame()
    names = []
    for ticker in tickers:
        try:
            # Fetching the adjusted close prices directly using yfinance
            temp_data = yf.download(ticker, start="2017-10-11", end="2024-10-11")['Adj Close']
            data = pd.concat([data, temp_data], axis=1)
            names.append(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    data.columns = names
    return data

# Ticker names of the companies that we will be looking at
ticks = ["GOOG", "NVDA", "META", "WFC", "DIS", "IMAX", "LUV", "PFE", "COKE", "CAT"]

# Fetch the data
d = get_historical_data(ticks)

# Check the data dimensions
print(d.shape)

# Most recent data
print(d.tail())


# In[3]:


# Saving the most recent year data such that we can compare...
# Called dT (DataTest)
dT = d.iloc[d.shape[0] - 252:,:] # Data test

# Update the "Training" or "data full"
d = d.iloc[:d.shape[0] - 252,:] # Data Train for the Simulation

print("Testing Data dimensions: ", dT.shape)
print("Training Data dimensions:", d.shape)


# In[4]:


dT # Test


# In[5]:


d # Train


# # Understanding Returns

# In[15]:


from scipy import stats
expected_returns_a = d.pct_change() # Daily returns from trading day to day...
expected_returns_a.columns = ticks # Setting the Column names 
expected_returns_aA = pd.DataFrame(expected_returns_a.mean()*250) # Annualizing the average rate of return
expected_returns_aA = expected_returns_aA.T # Transpose the values 

dar = d.pct_change().iloc[1:,:]+1 # dar = portfolio returns for each period (in this case day to day)
# 6 is the number of years I am working with (Note: Remember that earlier I've took out a year for training purposes.)
gar = pd.DataFrame(np.prod(dar)**(1/float(6)) - 1) # Geometric Average Rate of Return
full_return_annual = (pd.concat([expected_returns_aA.T, gar], axis = 1))
# DO NOTE that Arithmetic Average Return is not usually an appropriate method
# for calculating the average return and telling others...

# Example: Returns are the following (50%, 30%, -50%) on a yearly basis (jan 1st to dec 31st)
# Average: (50 + 30 - 50) / 3 = 10% average rate of return. This is not a great "representation of how well you done"
# Example
# Start with initial value of $ 100 Dollars: 
# First year becomes 150. 
# Second Year becomes 190. 
# Third year becomes 97.5. You LOST money.

# Geometric Average: (also known as the Compounded annual growth rate)
# Using the example from above...
# ((1+ 0.5) * (1 + 0.3) * (0.5))^(1/3) - 1
# ((1.5)*(1.3)*(0.5))^(1/3) - 1
# .9916 - 1
# -0.0084
# or (-0.84) % average ANNUAL rate of return (more accurate gauge as to how well you've done.)

full_return_annual.columns = ["Average Arithmetic Returns", "Average Geometric Returns"] 
print("Expected Annual Returns ", expected_returns_aA)
print("dar", dar)
print("Full Annual Return", full_return_annual)


# In[7]:


# Storing lists that retain returns, volatility, and weights of the Simulated portfolios
portfolio_returns = []
portfolio_volatility = []
sharpe_ratio = []

# This is what is going to be randomized
stock_weights = []

# Number of Indiviudal securities that will be a part of the portfolio
num_assets = len(ticks)
# Number of simulated iterations
num_portfolios = 100000

# Getting the covariance matrix
# Gets a percentage change one day to the next
daily_returns = d.pct_change()
# Converting daily returns to annual returns (standardizing to a year)
annual_returns = (daily_returns.mean() * 250) + 1

# Obtaining the covariance of annual
cov_daily = daily_returns.cov() # Covariance
cov_annual = cov_daily*250 # Covariance Annualized

print(annual_returns)


# In[8]:


# Setting seed of interpretability
np.random.seed(3)
# Filling in the lists with a simulated return, risk, and a given weight
# num_portfolios
for i in range(num_portfolios):
    # Randomly assign weights
    weights = np.random.random(num_assets)
    # Standardize the weights
    weights /= np.sum(weights)
    returns = (np.dot(weights, (annual_returns)))
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    """
    sharpe ratio: This calculates the risk adjusted return
    It suggests that adding assets to a portfolio that have low correlation can decrease portfolio risk without 
    sacrificing return 
    """
    sharpe = ((returns-1) / volatility)
    sharpe_ratio.append(sharpe)
    portfolio_returns.append(returns-1)
    portfolio_volatility.append(volatility)
    stock_weights.append(weights)


# In[9]:


# Storing the portfolio values
portfolio = {'Returns': portfolio_returns,
             'Volatility': portfolio_volatility,
             'Sharpe Ratio': sharpe_ratio}

# Add an additional entry to the portfolio such that each indivudal weight is incorporated for its corresponding company
for counter,symbol in enumerate(ticks):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)
df


# In[10]:


# PLotting the efficient frontier.
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()


# In[11]:


# Finding the Optimal Portfolio
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('fivethirtyeight')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()


# In[12]:


# Additional Details
r_ef = pd.concat([min_variance_port.T,sharpe_portfolio.T], axis = 1)
r_ef.columns = ["Minimum Risk Adjusted Values", "Max Risk Adjusted Values"]
print(r_ef)


# # If I were to invest 1,000 USD last year... what would I have now?

# In[13]:


amount_invest = 1000
expected_return = pd.DataFrame(amount_invest * (1+r_ef.iloc[0,:]))
print("----------------------------------------------------------------")
print("                Expected Returns on my Portfolio")
print("----------------------------------------------------------------")
print(expected_return.T)
print("")
print("----------------------------------------------------------------")
print("If I invested", amount_invest,"USD on |", dT.index[0],"| I would have...")
actual_return = (dT.iloc[dT.shape[0]-1,:] - dT.iloc[0,:]) / ( dT.iloc[0,:])
# Multipling the weights to the price at the beginning of the year
beg_price = (dT.iloc[0,:])
end_price = dT.iloc[dT.shape[0]-1,:]
print("----------------------------------------------------------------")
# Weights derived from the Efficient Frontier Portfolio
# Weights for Minimum Risk
w = np.array(r_ef.iloc[3:,0])

percentage_change = (end_price - beg_price)/(beg_price)+1
print("Using the Portfolio Weights for Minimum Risk Return Portfolio")
money_left = sum(w * percentage_change* amount_invest)
print("")
print("    Starting balance $ 1000 : Ending with $ ",round(money_left, 2))
print("")
print("----------------------------------------------------------------")
print("Using the Portfolio Weights Maximized Risk-Return Portfolio")
# Weights for Maxmimum Risk
w1 = np.array(r_ef.iloc[3:,1])

money_left1 = sum(w1 * percentage_change* amount_invest)
print("")
print("    Starting balance $ 1000 : Ending with $ ", round(money_left1,2))
print("")


# In[ ]:




