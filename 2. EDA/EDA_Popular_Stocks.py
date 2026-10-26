import numpy as np
import pandas as pd

df2017 = pd.read_excel('data_2017_for_sc.xlsx')
df2018 = pd.read_excel('data_2018_for_sc.xlsx')
df2019 = pd.read_excel('data_2019_for_sc.xlsx')
df2020 = pd.read_excel('data_2020_for_sc.xlsx')
df2021 = pd.read_excel('data_2021_for_sc.xlsx')

df2017_2020 = pd.concat([df2017, df2018, df2019, df2020, df2021])

A_df = df2017_2020[df2017_2020['SecurityType'] == 'A股']
A_df = A_df.groupby(A_df['ShortName'])['SecurityType'].value_counts()
A_df = pd.DataFrame(A_df)
A_df.columns = ['freq']
A_df = A_df.sort_values('freq', ascending=False)

writer = pd.ExcelWriter('stocks_freq.xlsx')
A_df.to_excel(writer)
writer.save()