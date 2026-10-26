import numpy as np
import pandas as pd

df2017 = pd.read_excel('data_2017_for_sc.xlsx', dtype={"Symbol":str}, engine='openpyxl')
df2018 = pd.read_excel('data_2018_for_sc.xlsx', dtype={"Symbol":str}, engine='openpyxl')
df2019 = pd.read_excel('data_2019_for_sc.xlsx', dtype={"Symbol":str}, engine='openpyxl')
df2020 = pd.read_excel('data_2020_for_sc.xlsx', dtype={"Symbol":str}, engine='openpyxl')
df2021 = pd.read_excel('data_2021_for_sc.xlsx', dtype={"Symbol":str}, engine='openpyxl')

# Popularity = ln(count(news on date t−1 of stock i）+1
Stock_freq2017 = df2017.groupby(df2017['Symbol'])['DeclareDate'].value_counts(sort=False)
Stock_freq2017 = pd.DataFrame(Stock_freq2017)
Stock_freq2017.columns = ['freq']
Stock_freq2017['Popularity'] = np.log(Stock_freq2017['freq'])+1  # +1 to avoid the news’s popularity from being disregarded
print(Stock_freq2017)

Stock_freq2018 = df2018.groupby(df2018['Symbol'])['DeclareDate'].value_counts(sort=False)
Stock_freq2018 = pd.DataFrame(Stock_freq2018)
Stock_freq2018.columns = ['freq']
Stock_freq2018['Popularity'] = np.log(Stock_freq2018['freq'])+1
print(Stock_freq2018)

Stock_freq2019 = df2019.groupby(df2019['Symbol'])['DeclareDate'].value_counts(sort=False)
Stock_freq2019 = pd.DataFrame(Stock_freq2019)
Stock_freq2019.columns = ['freq']
Stock_freq2019['Popularity'] = np.log(Stock_freq2019['freq'])+1
print(Stock_freq2019)

Stock_freq2020 = df2020.groupby(df2020['Symbol'])['DeclareDate'].value_counts(sort=False)
Stock_freq2020 = pd.DataFrame(Stock_freq2020)
Stock_freq2020.columns = ['freq']
Stock_freq2020['Popularity']= np.log(Stock_freq2020['freq'])+1
print(Stock_freq2020)

Stock_freq2021 = df2021.groupby(df2021['Symbol'])['DeclareDate'].value_counts(sort=False)
Stock_freq2021 = pd.DataFrame(Stock_freq2021)
Stock_freq2021.columns = ['freq']
Stock_freq2021['Popularity']= np.log(Stock_freq2021['freq'])+1
print(Stock_freq2021)

# save the new data into the excel
Popularity_2017 = []
for i in range(len(Stock_freq2017)):
    freq = int(Stock_freq2017.iloc[i]['freq'])
    pop = Stock_freq2017.iloc[i]['Popularity']
    for j in range(freq):
        Popularity_2017.append(pop)
df2017['Popularity'] = Popularity_2017
writer = pd.ExcelWriter('newdata_2017.xlsx')
df2017.to_excel(writer)
writer.save()

Popularity_2018 = []
for i in range(len(Stock_freq2018)):
    freq = int(Stock_freq2018.iloc[i]['freq'])
    pop = Stock_freq2018.iloc[i]['Popularity']
    for j in range(freq):
        Popularity_2018.append(pop)
df2018['Popularity'] = Popularity_2018
writer = pd.ExcelWriter('newdata_2018.xlsx')
df2018.to_excel(writer)
writer.save()

Popularity_2019 = []
for i in range(len(Stock_freq2019)):
    freq = int(Stock_freq2019.iloc[i]['freq'])
    pop = Stock_freq2019.iloc[i]['Popularity']
    for j in range(freq):
        Popularity_2019.append(pop)
df2019['Popularity'] = Popularity_2019
writer = pd.ExcelWriter('newdata_2019.xlsx')
df2019.to_excel(writer)
writer.save()

Popularity_2020 = []
for i in range(len(Stock_freq2020)):
    freq = int(Stock_freq2020.iloc[i]['freq'])
    pop = Stock_freq2020.iloc[i]['Popularity']
    for j in range(freq):
        Popularity_2020.append(pop)
df2020['Popularity'] = Popularity_2020
writer = pd.ExcelWriter('newdata_2020.xlsx')
df2020.to_excel(writer)
writer.save()

Popularity_2021 = []
for i in range(len(Stock_freq2021)):
    freq = int(Stock_freq2021.iloc[i]['freq'])
    pop = Stock_freq2021.iloc[i]['Popularity']
    for j in range(freq):
        Popularity_2021.append(pop)
df2021['Popularity'] = Popularity_2021
writer = pd.ExcelWriter('newdata_2021.xlsx')
df2021.to_excel(writer)
writer.save()