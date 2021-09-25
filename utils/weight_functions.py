from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return


def historical_returns(history_df):
    return history_df.pct_change().cumsum().fillna(0)


def uniform_weight_returns(history_df, _):
    u = 1 / len(history_df.columns)
    return { asset: u for asset in history_df.columns }


def HRP_weight(history_df, _):
    returns = historical_returns(history_df)
    optimizer = HRPOpt(returns=returns).optimize()
    return optimizer


def CLA_weight(history_df, _):
    returns = ema_historical_return(history_df)
    optimizer = CLA(expected_returns=returns, cov_matrix=history_df.cov())
    return optimizer


def MVO_weight(history_df, model_config):
    returns = ema_historical_return(history_df)
    ef = EfficientFrontier(returns, history_df.cov(), verbose=True)
    try:
        weights = uniform_weight_returns(history_df, model_config)
        if model_config.optimize_method == 'volatility':
            weights = ef.min_volatility()
        elif model_config.optimize_method == 'sharpe':
            weights = ef.max_sharpe(model_config.risk_free_rate)
        elif model_config.optimize_method == 'risk':
            weights = ef.efficient_risk(model_config.target_volatility, model_config.market_neutral)
        elif model_config.optimize_method == 'return':
            weights = ef.efficient_return(model_config.target_return, model_config.market_neutral)
        return weights
    except:
        return uniform_weight_returns(history_df, model_config)