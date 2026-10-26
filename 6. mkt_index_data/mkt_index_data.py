import pandas as pd

# index data to be used in trading strategy
index_data = pd.read_excel('TRD_Index.xlsx').drop(labels = [0,1]).reset_index(drop = True)
index_data.rename(columns={'Trddt':'Trading_Date','Daywk':'Dayofweek','Opnindex':'Open_index',
                          'Hiindex':'High_index','Loindex':'Low_index','Clsindex':'Close_index',
                          'Retindex':'Return_ratio'},inplace=True) # rename the columns for convenience
index_data.drop(['Dayofweek'],axis=1,inplace=True)
print(index_data.head())

# keep the columns for potential use
index_data.drop(['High_index','Low_index','Return_ratio'],axis=1,inplace=True)
index_data['interest_riskfree_day'] = 0.000041 # daily risk-free interest rate (deposit and withdraw in lump sum for one-year time deposit)
index_data['interest_riskfree_year'] = 0.015 # yearly risk-free interest rate (deposit and withdraw in lump sum for one-year time deposit)
index_data['Return_Ratio_t'] = (index_data['Close_index']-index_data['Open_index'])/index_data['Open_index'] # calculate the return ratio on each trading day

# choose four kinds of index data
shangzhen_a = index_data[index_data['Indexcd']== '000002'] # A-shares in Shanghai market
shenzheng_a = index_data[index_data['Indexcd']== '399107'] # A-shares in Shenzhen market
shangzhen_180 = index_data[index_data['Indexcd']== '000010'] # Shanghai SSE 180 Index
hushen_300 = index_data[index_data['Indexcd']== '000300'] # Shanghai Shenzhen CSI 300 Index