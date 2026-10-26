import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.options.display.max_columns = None
pd.options.display.width = None

# Firstly, calculate the index return ratio -- Benchmark_1
hushen = pd.read_excel('hushen.xlsx')
hushen_year = hushen
hushen_year['date'] = pd.to_datetime(hushen_year['Trading_Date'])
hushen_year = hushen_year.set_index('date')
hushen_year['Return'] = (hushen_year['Close_index']-hushen_year['Open_index'])/hushen_year['Close_index']
hushen_year = hushen_year['2018':'2021']  # change year
hushen_year['excess_return'] = hushen_year['Return'] - hushen_year['interest_riskfree_day']

avg_index_all = hushen_year['excess_return'].mean()
std_index_all = hushen_year['excess_return'].std()
sharp_index_all = avg_index_all/std_index_all
maxdrawdown_index_all = np.min([np.sum(hushen_year['Return'][i:j]) for i in range(len(hushen_year['Return'])) for j in range(i, len(hushen_year['Return']))])
calmar_index_all = avg_index_all/abs(maxdrawdown_index_all)


# Our strategy part
# data processing
test18 = pd.read_csv('data_2018_pred.csv')
test19 = pd.read_csv('data_2019_pred.csv')
test20 = pd.read_csv('data_2020_pred.csv')
test21 = pd.read_csv('data_2021_pred.csv')
test = pd.concat([test18,test19,test20,test21],ignore_index=True)

test['stmt_pred_wa'] = 2*test['pred_label_0']+test['pred_label_1']  # calculate the weighted sentiment score
test['stmt_pred_popula'] = test['Popularity_log'] * test['stmt_pred_wa'] # calculate sentiment score * populatity
test['return'] = (test['Close_Price']-test['Open_Price'])/test['Open_Price'] # calculate the return c-o/o

# buy which stocks
# buy sotcks with max 5 weighted avg sentiment score
date_lis = test.Trading_Date.unique()
test['buy_b2'] = None
for i in date_lis:
    lis = list(test.loc[test.Trading_Date == i].sort_values('stmt_pred_wa', ascending=False)[:5].index)
    test.buy_b2[lis] = 1
sentiment_order = test.loc[test.buy_b2 == 1]

# buy stocks with classifier model
model_stra = test.loc[test.ylabel_pred == 1]

# buy stocks whose weighted avg sentiment score less than 1 (0-positive,1-nural,2-negative)
sentiment_all = test.loc[test.stmt_pred == 1]

# buy stocks with max 5 weighted avg sentiment scor*popularity
test['buy_sp'] = None
for i in date_lis:
    lis = list(test.loc[test.Trading_Date == i].sort_values('stmt_pred_popula', ascending=False)[:5].index)
    test.buy_sp[lis] = 1
sentiment_popularity = test.loc[test.buy_sp == 1]

# extract the ratio of different strategy
def return_lis(stra_port,hushen_year):
    stra_port = stra_port.sort_values('Trading_Date')
    series = stra_port.groupby('Trading_Date')['return'].mean()
    date = list(series.index)
    r_rate = list(series.values)
    df_test = pd.DataFrame({'Trading_Date':date, 'portfolio_return':r_rate})

    df_final = pd.merge(df_test,hushen_year,on='Trading_Date')
    df_final['excessR_port'] = df_final['portfolio_return'] - df_final['interest_riskfree_day']
    df_final['excessR_index'] = df_final['Return'] - df_final['interest_riskfree_day']
    df_final['excess_return'] = df_final['portfolio_return'] - df_final['Return']

    plt.plot(df_final.excess_return.cumsum())
    plt.show()

    # Calculate different standards
    # new sharp ratios

    # sharp ratio
    sharp_ratio_port = df_final['excessR_port'].mean()/df_final['excessR_port'].std()
    sharp_ratio_index = df_final['excessR_index'].mean()/df_final['excessR_index'].std()

    # average return
    avg_port = df_final['portfolio_return'].mean()
    avg_index = df_final['Return'].mean()

    # standard deviation
    std_port = df_final['portfolio_return'].std()
    std_index = df_final['Return'].std()

    # max draw down
    maxdrawdown_port = np.min([np.sum(df_final['portfolio_return'][i:j]) for i in range(len(df_final['portfolio_return'])) for j in range(i, len(df_final['portfolio_return']))])
    maxdrawdown_index = np.min([np.sum(df_final['Return'][i:j]) for i in range(len(df_final['Return'])) for j in range(i, len(df_final['Return']))])

    # calmar ratio
    calmar_port = avg_port/abs(maxdrawdown_port)
    calmar_index = avg_index/abs(maxdrawdown_index)

    print('Sharp ratio for our strategy and index are: {} and {}'.format(sharp_ratio_port,sharp_ratio_index))
    print('Average return for our strategy and index are: {} and {}'.format(avg_port,avg_index))
    print('Standard deviation for our strategy and index are: {} and {}'.format(std_port,std_index))
    print('Max draw down for our strategy and index are: {} and {}'.format(maxdrawdown_port,maxdrawdown_index))
    print('Calmar ration for our strategy and index are: {} and {}'.format(calmar_port,calmar_index))
    return [round(sharp_ratio_port,5),round(avg_port,5),round(std_port,5),round(maxdrawdown_port,5),round(calmar_port,5)]

# result table
name = ['sharp_ratio','average_return','standard_deviation','maxdrawdown','calmar_ratio']
index_lis = [round(sharp_index_all,5),round(avg_index_all,5),round(std_index_all,5),round(maxdrawdown_index_all,5),round(calmar_index_all,5)]
sentiment_order_lis = return_lis(sentiment_order,hushen_year)
model_stra_lis = return_lis(model_stra,hushen_year)
sentiment_all_lis = return_lis(sentiment_all,hushen_year)
sentiment_popularity = return_lis(sentiment_popularity,hushen_year)

result_dic = {'index':index_lis,'sentimen_order':sentiment_order_lis,'model_strategy':model_stra_lis,
              'sentiment_all':sentiment_all_lis,'sentiment_populatity':sentiment_popularity}
result = pd.DataFrame(result_dic)
result = result.T
result.columns = name
print(result)










