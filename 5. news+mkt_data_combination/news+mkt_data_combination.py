import pandas as pd
import warnings
warnings.simplefilter("ignore")
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

# popularity indicators
popularity_log = []
for i in range(2017,2022):
    p_log_df = pd.read_excel('newdata_{}.xlsx'.format(i),dtype={"Symbol":str},engine='openpyxl') # output files of popularity calculation (log+1)
    p_log_keep = p_log_df[['DeclareDate','Symbol','ShortName','SecurityType','Popularity']]
    p_log_keep.rename(columns = {'Popularity':'Popularity_log'},inplace=True)
    p_log_keep.drop_duplicates(inplace=True)
    popularity_log.append(p_log_keep)
p_log_total = pd.concat(popularity_log,ignore_index=True) # combine data for all years
p_log_total.drop_duplicates(subset=['DeclareDate','Symbol'],inplace=True)
p_log_total.drop_duplicates(subset=['DeclareDate','ShortName'],inplace=True)

# choose A-shares stocks as the target
p_total = p_log_total[p_log_total['SecurityType'] == 'A股']


# sentiment score
for i in range(2017,2022):
    ss = pd.read_csv('res_lstm_{}.csv'.format(i),converters={"Symbol":str}) # output files of sentiment scores
    ss.drop_duplicates(subset=['NewsID','Symbol'], inplace = True)
    ss = ss[ss['SecurityType'] == 'A股']
    ss = ss[['DeclareDate','ShortName','Symbol','pred_label']]
    ss_dummy = pd.get_dummies(ss['pred_label'], prefix = 'pred_label') # convert sentiment scores into dummy variable
    ss = pd.concat([ss,ss_dummy], axis= 1)
    ss.drop(['pred_label'],axis=1, inplace=True)
    ss = ss.groupby(['DeclareDate','Symbol','ShortName'], as_index=False).mean() # do aggregations (mean) so that each dummy variable could represent the degree of sentiment
    ss.sort_values(by=['Symbol','DeclareDate'],inplace = True)
    ss.drop(['Symbol'],axis=1,inplace=True)
    ss['DeclareDate'] = pd.to_datetime(ss['DeclareDate'],format='%Y-%m-%d')
    p_total['DeclareDate'] = pd.to_datetime(p_total['DeclareDate'],format='%Y-%m-%d')
    p_year = p_total[p_total['DeclareDate'].dt.year == i]
    df_year = pd.merge(p_year,ss,on=['DeclareDate','ShortName'],how='left') # combine the popularity and sentiment score
    df_year.dropna(inplace=True)
    df_year.to_csv('p_ss_{}_dummy_a.csv'.format(i),index=False)
    print('Year {} is done!'.format(i))

# marketing transaction data gained from CSMAR
mkt_data_list = []
for i in range(2016,2022):
    if i < 2021:
        mkt_data = pd.read_excel('TRD_Dalyr_{}.xlsx'.format(i)).drop(labels = [0,1]).reset_index(drop = True)
    else:
        mkt_data_1 = pd.read_excel('TRD_Dalyr_{}_1.xlsx'.format(i)).drop(labels = [0,1]).reset_index(drop = True)
        mkt_data_2 = pd.read_excel('TRD_Dalyr_{}_2.xlsx'.format(i)).drop(labels = [0,1]).reset_index(drop = True)
        mkt_data = pd.concat([mkt_data_1,mkt_data_2],ignore_index=True)  # year 2021 has two separate files
    mkt_data_list.append(mkt_data)
    print('Year {} is done!'.format(i))
mkt_data_total = pd.concat(mkt_data_list,ignore_index=True)
mkt_data_sort = mkt_data_total.sort_values(by=['Stkcd','Trddt'] , axis=0, ascending=[True,True], inplace=False) # sort by stock symbol and transaction date
mkt_data_sort = mkt_data_sort.reset_index(drop = True)

# rolling one trading day for the close price
mkt_data_sort['T-1_Clsprc'] = mkt_data_sort.groupby('Stkcd')['Clsprc'].shift(1)

# gain some historical transaction indicators for model building
mkt_data_sort['T_H-L/C'] = (mkt_data_sort['Hiprc'] - mkt_data_sort['Loprc'])/mkt_data_sort['Clsprc'] # (high price - low price)/close price, do scale reduction
mkt_data_sort['T-1_H-L/C'] = mkt_data_sort.groupby('Stkcd')['T_H-L/C'].shift(1) # rolling one trading day

mkt_data_sort['T_C-O/C'] = (mkt_data_sort['Clsprc'] - mkt_data_sort['Opnprc'])/mkt_data_sort['Clsprc'] # (close price - open price)/close price, do scale reduction
mkt_data_sort['T-1_C-O/C'] = mkt_data_sort.groupby('Stkcd')['T_C-O/C'].shift(1) # also rolling one trading day


# rolling 7/14/21 days for moving avg of stock prices and do scale reduction
mkt_data_sort['rolling_7_days_avg'] = mkt_data_sort.groupby('Stkcd')['Clsprc'].transform(
    lambda s: s.rolling(window=7, min_periods=7).mean())
mkt_data_sort['rolling_7_days_avg'] = mkt_data_sort.groupby('Stkcd')['rolling_7_days_avg'].shift(1)
mkt_data_sort['rolling_7_days_avg'] = mkt_data_sort['rolling_7_days_avg']/mkt_data_sort['T-1_Clsprc']
mkt_data_sort['rolling_14_days_avg'] = mkt_data_sort.groupby('Stkcd')['Clsprc'].transform(
    lambda s: s.rolling(window=14, min_periods=14).mean())
mkt_data_sort['rolling_14_days_avg'] = mkt_data_sort.groupby('Stkcd')['rolling_14_days_avg'].shift(1)
mkt_data_sort['rolling_14_days_avg'] = mkt_data_sort['rolling_14_days_avg']/mkt_data_sort['T-1_Clsprc']
mkt_data_sort['rolling_21_days_avg'] = mkt_data_sort.groupby('Stkcd')['Clsprc'].transform(
    lambda s: s.rolling(window=21, min_periods=21).mean())
mkt_data_sort['rolling_21_days_avg'] = mkt_data_sort.groupby('Stkcd')['rolling_21_days_avg'].shift(1)
mkt_data_sort['rolling_21_days_avg'] = mkt_data_sort['rolling_21_days_avg']/mkt_data_sort['T-1_Clsprc']


# rolling 7 days for std of stock prices
mkt_data_sort['rolling_7_days_std'] = mkt_data_sort.groupby('Stkcd')['Clsprc'].transform(
    lambda s: s.rolling(window=7, min_periods=7).std())
mkt_data_sort['rolling_7_days_std'] = mkt_data_sort.groupby('Stkcd')['rolling_7_days_std'].shift(1)

# rename the columns for convenience
mkt_data_sort.rename(columns = {'Stkcd':'Symbol','Trddt':'Trading_Date','Opnprc':'Open_Price',
                               'Hiprc':'High_Price','Loprc':'Low_Price','Clsprc':'Close_Price',
                               'Dnshrtrd':'Daily_trading_units','Dnvaltrd':'Daily_trading_amt',
                               'Dsmvosd':'Daily_circulated_value','Dsmvtll':'Daily_total_value',
                               'Dretwd':'Return_rate_b','Dretnd':'Return_rate_nb'},inplace = True)

# create the day before for each trading day, to be joined with popularity and sentiment score
mkt_data_sort['Trading_Date'] = pd.to_datetime(mkt_data_sort['Trading_Date'],format='%Y-%m-%d')
mkt_data_sort['Day_before_trading_date'] = mkt_data_sort['Trading_Date'] + datetime.timedelta(days = -1)

# create y_label to indicate whether the stock price will rise during each trading day
y_label = []
for i in range(mkt_data_sort.shape[0]):
    if mkt_data_sort.Close_Price[i] > mkt_data_sort.Open_Price[i]:
        y_label.append(1)
    else:
        y_label.append(0)
mkt_data_sort['y_label'] = y_label

for i in range(2017,2022):
    mkt_data_sort[mkt_data_sort['Trading_Date'].dt.year == i].to_csv('mkt_data_{}.csv'.format(i),index=False) # export the mkt data
    print('Marketing data for year {} is done!'.format(i))


# final data combination： popularity+sentiment score+mkt data
for i in range(2017,2022):
    mkt_data = pd.read_csv('mkt_data_{}.csv'.format(i),converters={'Symbol':str})
    mkt_data.rename(columns={'Day_before_trading_date':'DeclareDate'},inplace=True)
    mkt_data.drop(['Ahshrtrd_D','Ahvaltrd_D'],axis=1,inplace=True)
    p_ss_data = pd.read_csv('p_ss_{}_dummy_a.csv'.format(i),converters={'Symbol':str})
    data_merge = pd.merge(mkt_data,p_ss_data,how='left',on=['Symbol','DeclareDate'])
    data_merge.dropna(inplace=True)
    data_merge.to_csv('all_data_year_{}_a.csv'.format(i),index=False)
    print('Data for year {} is done!'.format(i))

print(data_merge.columns)

