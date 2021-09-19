from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt


def historical_returns(history_df):
    return history_df.pct_change().cumsum().fillna(0)


def HRP_weight(history_df):
    returns = historical_returns(history_df)
    optimizer = HRPOpt(returns=returns).optimize()
    return optimizer


def CLA_weight(history_df):
    # TODO: convert to expected
    returns = historical_returns(history_df)
    optimizer = CLA(expected_returns=returns)
    return optimizer


def MVO(history_df):
    returns = historical_returns(history_df)
    pass