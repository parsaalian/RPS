import pandas as pd


def SP500_loader(filename, start_date=None, end_date=None):
    history = pd.read_history(filename)
    history.columns= history.columns.str.lower()

    if start_date is not None:
        history = history[history['date'] >= start_date] 
    if end_date is not None:
        history = history[history['date'] <= end_date]
    
    history = history.set_index(['date', 'name'])['close'].unstack(-1)
    history.index = pd.to_datetime(history.index)
    
    for stock in history.columns:
        if (history[stock].isnull().values.any()):
            history.drop(columns=[stock])

    history = history.fillna(method='ffill')
    
    return history