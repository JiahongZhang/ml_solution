import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
## TODO: local data report

plt.rcParams["axes.unicode_minus"] = False # 该语句解决图像中的“-”负号的乱码问题



def append_single_row_dict(df, row_dict):
    row = pd.DataFrame([row_dict])
    df = pd.concat([df, row], ignore_index=True)
    return df



def series_count(
        series, 
        draw_bar=True, 
        percent_sum_th=70, 
        percent_th=3,
        min_other_rows=10,
        not_show_table_rows=30
        ):
    name = series.name
    df = series.value_counts().reset_index()
    df['percent'] = round(100*df['count']/df['count'].sum(), 2)
    missing_num = series.isna().sum()
    na_row = {
        name: 'Missing', 
        'count': missing_num, 
        'percent': round(100*missing_num/len(series), 2)
    }

    if not draw_bar:
        df = append_single_row_dict(df, na_row)
        return df
    
    df_draw = df.copy()
    if len(df)>min_other_rows:
        p_sum = 0
        for i, p in enumerate(df_draw['percent']):
            # print(i, p)
            p_sum += p
            if p_sum>=percent_sum_th or p<=percent_th:
                break

        other_row = {
            name: 'Others', 
            'count': df_draw['count'].iloc[i+1:].sum(), 
            'percent': df_draw['percent'].iloc[i+1:].sum()
        }
        df_draw.drop(df_draw.index[i+1:], inplace=True)
        df_draw = append_single_row_dict(df_draw, other_row)

    fig = plt.figure(dpi=300)
    palette_color = sns.color_palette("muted")
    plt.title(name) 
    plt.pie(df_draw['count'], labels=df_draw[name], \
            colors=palette_color, autopct='%.0f%%') 

    df = append_single_row_dict(df, na_row)
    df.rename(columns={name: 'keys'}, inplace=True)
    if len(df)<=not_show_table_rows:
        plt.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            bbox=(1.1, 0, 0.5, 0.05*len(df))
            )

    
    return df, fig








