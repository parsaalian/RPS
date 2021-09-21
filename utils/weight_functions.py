from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.expected_returns import ema_historical_return


def historical_returns(history_df):
    return history_df.pct_change().cumsum().fillna(0)


def HRP_weight(history_df):
    returns = historical_returns(history_df)
    optimizer = HRPOpt(returns=returns).optimize()
    return optimizer


def CLA_weight(history_df):
    returns = ema_historical_return(history_df)
    optimizer = CLA(expected_returns=returns, cov_matrix=history_df.cov())
    return optimizer


def MVO(history_df):
    returns = historical_returns(history_df)
    pass