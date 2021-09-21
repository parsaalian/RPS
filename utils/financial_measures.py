import numpy as np
from scipy.stats import kurtosis, skew, linregress
from utils.constants import RISK_FREE_RATE, DAYS_IN_YEAR


def calculate_returns(prices, assets, weights):
    asset_prices = prices[assets]
    return asset_prices.pct_change(len(asset_prices) - 1).dropna().values[0] * weights


def modigliani_ratio(sharpe_ratio, benchmark_returns):
    benchmark_volatility = benchmark_returns.std() * np.sqrt(DAYS_IN_YEAR)
    m2_ratio = (sharpe_ratio * benchmark_volatility) + RISK_FREE_RATE
    return m2_ratio


def treynor_ratio(port_return ,port_returns, benchmark_returns):
    return (port_return - RISK_FREE_RATE) / beta(port_returns, benchmark_returns)


def adjusted_sharpe_ratio(prices, assets, weights):
    _, _, sharpe = sharpe_ratio(weights, assets, prices)
    returns = calculate_returns(prices, assets, weights)
    return sharpe * (1 + skew(returns) * sharpe / 6 +  (kurtosis(returns) - 3) * (sharpe ** 2) / 24)


def information_ratio(port_returns, benchmark_returns):
    return_difference = (port_returns - benchmark_returns).dropna()
    volatility = return_difference.std() * np.sqrt(DAYS_IN_YEAR)
    information_ratio = return_difference.mean() / volatility
    return information_ratio


def sharpe_ratio(port_return, port_std):
    return ((port_return - RISK_FREE_RATE) / port_std)


def beta(port_returns, benchmark_returns):
    # benchmark_returns and port_returns do not have the same len sometimes
    # some price data may be missing
    return linregress(benchmark_returns.dropna().values,port_returns[:len(benchmark_returns)].dropna().values).slope


def calculate_measures(stocks, price_df, weights):
    print(1)
    if len(stocks) == 0:
        return ([], 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    print(2)
    stocks_prices = price_df[stocks].fillna(0, axis=1)
    
    print(3)
    port_std = np.sqrt(
        np.dot(weights, np.dot(stocks_prices.pct_change(1).dropna().cov().values, weights))
    ) * np.sqrt(DAYS_IN_YEAR)
    port_returns = (stocks_prices.pct_change().dropna(how='all') * weights).sum(axis=1)
    port_return = port_returns.mean() * 252
    benchmark_returns = (price_df.pct_change() * ([1/len(price_df.columns)]*len(price_df.columns))).sum(axis=1)
    
    print(4)
    port_sharpe = sharpe_ratio(port_return, port_std)
    port_information = information_ratio(port_return, benchmark_returns)
    port_modigliani = modigliani_ratio(port_sharpe, benchmark_returns)
    # port_treynor = treynor_ratio(port_return, port_returns, benchmark_returns)
    
    print(5)
    corrs = price_df.corr().stack().reset_index(level=0).rename(
        columns={'name':'name1'}
    ).reset_index().rename(
        columns={'name':'name2', 0: 'corr'}
    )
    corrs = corrs[corrs['name1'] != corrs['name2']]
    desc = corrs[corrs.name1.isin(stocks) & corrs.name2.isin(stocks)]['corr'].describe()
    corr_min, corr_max, corr_mean, corr_std = desc['min'], desc['max'], desc['mean'], desc['std']
    
    print(6)
    
    return (
        weights,
        corr_min,
        corr_max,
        corr_mean,
        corr_std,
        port_return,
        port_std,
        port_sharpe,
        port_information,
        port_modigliani
        # port_treynor
    )