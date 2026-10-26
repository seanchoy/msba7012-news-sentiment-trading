import pandas as pd
import warnings
warnings.simplefilter("ignore")

# combine the news-class data
news_class_list = []
for i in range(2,5):
    news_class = pd.read_excel('News_Classify{}.xlsx'.format(i)).drop(labels = [0,1]).reset_index(drop = True) # [raw data provided by the instructor] files containing the news class info between 2017 and 2021
    news_class_list.append(news_class)
    print('File {} is done!'.format(i))
news_class_total = pd.concat(news_class_list,ignore_index=True) # combine the files above
news_class_total = news_class_total[['NewsID','ClassifyID','ClassifyName']] # keep the columns for potential use
print(news_class_total.shape)


# combine the news-security data, here we rename the files to facilitate reading
news_stocks_2017_2020 = pd.read_excel('News_Security_2017_2020.xlsx').drop(labels = [0,1]).reset_index(drop = True)
news_stocks_2021 = pd.read_excel('News_Security_2021.xlsx').drop(labels = [0,1]).reset_index(drop = True) # [raw data provided by the instructor] files containing news-stock connections
news_stocks_total = pd.concat([news_stocks_2017_2020,news_stocks_2021],ignore_index=True) # combine the files above
news_stocks_total=news_stocks_total[['NewsID','Symbol','ShortName','SecurityTypeID','SecurityType']] # keep the columns for potential use
print(news_stocks_total.shape)

# combine all the data provided by the instructor
for i in range(2017,2022):
    news_info = pd.read_excel('News_NewsInfo_{}.xlsx'.format(i)).drop(labels = [0,1]).reset_index(drop = True) # [raw data provided by the instructor] files containing the news basic info, here we also rename the files
    news_info_class = pd.merge(news_info,news_class_total,how='left',on='NewsID') # merge news info and news class
    news_info_class['KeyWord'] = news_info_class['KeyWord'].fillna('')
    news_info_class['NewsSummary'] = news_info_class['NewsSummary'].fillna('')
    news_info_class = news_info_class.drop(['ClassifyID','NewsSourceID','Autor'],axis=1)
    news_info_class_stocks = pd.merge(news_info_class,news_stocks_total,how='left',on='NewsID') # final data set with news basic info, class, matched stock
    news_info_class_stocks.dropna(subset=['Symbol'], inplace=True) # drop the rows without matched stocks
    news_info_class_stocks = news_info_class_stocks.sort_values(by = ['Symbol','DeclareDate']).reset_index(drop = True) # sort the data by stock symbol and the declare date of news for visual convenience
    news_info_class_stocks.drop_duplicates(subset=['NewsID','Symbol'], inplace = True) # check and drop duplicates with newsid and symbol
    news_info_class_stocks.to_excel('data_{}_for_sc.xlsx'.format(i),index=False) # export files for sentiment analysis and popularity calculation
    print('Data for year {} is done!'.format(i))


