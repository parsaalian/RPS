import pandas as pd


def SP500_loader(dataset_config, run_config):
    history = pd.read_csv(dataset_config.data_path)
    history.columns= history.columns.str.lower()
    
    history['date'] = pd.to_datetime(history['date'])
    
    if run_config.start_date is not None:
        history = history[history['date'] >= run_config.start_date]
    if run_config.end_date is not None:
        history = history[history['date'] <= run_config.end_date]
    
    history = history.set_index(['date', 'name'])['close'].unstack(-1)
    
    for stock in history.columns:
        if (history[stock].isnull().values.any()):
            history.drop(columns=[stock])

    history = history.fillna(method='ffill')
    
    return history
