import numpy as np
import pandas as pd
from utils.readable_df import *
from utils.financial_measures import *


def calculate_future_performance(history_df, stocks, weights):
    return calculate_measures(stocks, history_df, weights)


def df_future_performance(
    history_df,
    weights_df,
    columns,
    save_path,
):  
    future_performances = []
    
    for _, row in weights_df.iterrows():
        future_performances.append([
            row.stocks,
            *calculate_measures(row.stocks, history_df, row.weights)
        ])
    
    future_performances = np.asarray(future_performances, dtype=object)
            
    df = pd.DataFrame({
        columns[i]: future_performances[:, i] for i in range(len(columns))
    })
    df = df_list_to_readable(df, ['stocks', 'weights'])
    df.to_csv(save_path, index=False)


def calculate_noise_stability(set_a, set_b):
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def calculate_time_stability():
    pass


def calculate_frontier():
    pass