import signal
import numpy as np
from contextlib import contextmanager

from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def historical_returns(history_df):
    returns = history_df.pct_change().cumsum().fillna(0)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def uniform_weight_returns(history_df, _):
    u = 1 / len(history_df.columns)
    return { asset: u for asset in history_df.columns }


def HRP_weight(history_df, _):
    returns = historical_returns(history_df).reset_index(drop=True)
    optimizer = HRPOpt(returns=returns).optimize()
    return optimizer


def CLA_weight(history_df, model_config):
    returns = ema_historical_return(history_df)
    with time_limit(1):
        optimizer = CLA(expected_returns=returns, cov_matrix=history_df.cov())
        if model_config.model_config == 'volatility':
            weights = optimizer.min_volatility()
        elif model_config.model_config == 'sharpe':
            weights = optimizer.max_sharpe()
        return weights


def MVO_weight(history_df, model_config):
    returns = ema_historical_return(history_df)
    ef = EfficientFrontier(returns, history_df.cov())
    with time_limit(1):
        weights = None
        if model_config.model_config == 'volatility':
            weights = ef.min_volatility()
        elif model_config.model_config == 'sharpe':
            weights = ef.max_sharpe(0.02)
        elif model_config.model_config == 'risk':
            weights = ef.efficient_risk(1.0, False)
        elif model_config.model_config == 'return':
            weights = ef.efficient_return(1.0, False)
        return weights