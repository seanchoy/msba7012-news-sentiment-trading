import pandas as pd

news_df2017 = pd.read_excel('data_2017_for_sc.xlsx')
news_df2018 = pd.read_excel('data_2018_for_sc.xlsx')
news_df2019 = pd.read_excel('data_2019_for_sc.xlsx')
news_df2020 = pd.read_excel('data_2020_for_sc.xlsx')
news_df2021 = pd.read_excel('data_2021_for_sc.xlsx')
news_df = pd.concat([news_df2017,news_df2018,news_df2019,news_df2020,news_df2021],axis=0)

writer = pd.ExcelWriter('data_2017_2022.xlsx')
news_df.to_excel(writer)
writer.save()

print(len(news_df['NewsSource'].unique()))

print(news_df['NewsSource'].unique())






