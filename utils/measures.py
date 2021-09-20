from scipy.spatial import distance
from utils.financial_measures import *


def calculate_future_performance(history_df, stocks, weights):
    return calculate_measures(stocks, history_df, weights)


def calculate_noise_stability(set_a, set_b):
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def calculate_time_stability():
    pass


def calculate_frontier():
    pass