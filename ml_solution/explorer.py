import pandas as pd



def series_count(series):
    df = series.value_counts().reset_index()
    df['percent'] = round(100*df['count']/df['count'].sum(), 2)
    return df








