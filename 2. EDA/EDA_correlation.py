import matplotlib.pyplot as plt
import pandas as pd
import csv

# read the csv
# take 2017 as sample
tmp_lst = []
with open('all_data_year_2017.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])

# pre-processing
df['pred_label_0'] = pd.to_numeric(df['pred_label_0'])
df['pred_label_1'] = pd.to_numeric(df['pred_label_1'])
df['pred_label_2'] = pd.to_numeric(df['pred_label_2'])
df['Popularity_log'] = pd.to_numeric(df['Popularity_log'])
df['y_label'] = pd.to_numeric(df['y_label'])
# get the weighted sentiment score
df['pred_label'] = (0*df['pred_label_0']+1*df['pred_label_1']+2*df['pred_label_2'])/3
df['SS*POP'] = df['pred_label']*df['Popularity_log']

# get the symbol list
Symbol_cor = list(set(df['Symbol'].tolist()))

# get the correlation of sentiment score and price trend
correlation_ss = []
for i in range(len(Symbol_cor)):
    symbol = Symbol_cor[i]
    cor = df.loc[df['Symbol'] ==Symbol_cor[i]]['y_label'].corr(df.loc[df['Symbol'] == Symbol_cor[i]]['pred_label'])
    correlation_ss.append([symbol,cor])

correlation_ss = pd.DataFrame(correlation_ss)
correlation_ss.columns = ['Symbol','cor_sen']
correlation_ss.sort_values(by='cor_sen',inplace=True, ascending=False)
print('print out the correlation of sentiment score and price trend')
print(correlation_ss)

# get the correlation of sentiment score*popularity and price trend
correlation_sspop = []
for i in range(len(Symbol_cor)):
    symbol = Symbol_cor[i]
    cor_sspop = df.loc[df['Symbol']==Symbol_cor[i]]['y_label'].corr(df.loc[df['Symbol'] == Symbol_cor[i]]['SS*POP'])
    correlation_sspop.append([symbol,cor_sspop])

correlation_sspop = pd.DataFrame(correlation_sspop)
correlation_sspop.columns = ['Symbol','cor_sen*pop']
correlation_sspop.sort_values(by='cor_sen*pop',inplace=True, ascending=False)
print('print out the correlation of sentiment score*popularity and price trend')
print(correlation_sspop)

correlation_final = pd.merge(correlation_ss,correlation_sspop,on='Symbol')
correlation_final = correlation_final.loc[(correlation_final['cor_sen'] > 0)&(correlation_final['cor_sen*pop'] > 0)]
correlation_final = correlation_final.sort_values(by='cor_sen*pop', ascending=False)
print('print out the final dataframe')
print(correlation_final)

# visualize some sample stocks
# 平安银行（000001）
plt.rcParams['figure.figsize']=(12.8, 7.2)
fig,ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
p1, = ax.plot(df.loc[df['Symbol'] == '000001']['Trading_Date'], df.loc[df['Symbol'] == '000001']['y_label'], "bo", label="y_label",linestyle = '-')
p2, = twin1.plot(df.loc[df['Symbol'] == '000001']['Trading_Date'],df.loc[df['Symbol'] == '000001']['pred_label'], "ro", label="pred_label",linestyle = '--')
ax.set_xlabel("Trading Date")
ax.set_ylabel("Price Change")
twin1.set_ylabel("Sentiment score")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(),**tkw)
twin1.tick_params(axis='y', colors=p2.get_color(),**tkw)
ax.tick_params(axis='x')
ax.legend(handles=[p1, p2])
plt.show() # show the correlation plot of sentiment score and price trend

# 平安银行（000001）
plt.rcParams['figure.figsize']=(12.8, 7.2)
fig,ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
p1, = ax.plot(df.loc[df['Symbol'] == '000001']['Trading_Date'], df.loc[df['Symbol'] == '000001']['y_label'], "bo", label="y_label",linestyle = '-')
p2, = twin1.plot(df.loc[df['Symbol'] == '000001']['Trading_Date'],df.loc[df['Symbol'] == '000001']['SS*POP'], "ro", label="Sentiment_score*Popularity_log",linestyle = '--')
ax.set_xlabel("Trading Date")
ax.set_ylabel("Price Change")
twin1.set_ylabel("Sentiment_score*Popularity_log")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(),**tkw)
twin1.tick_params(axis='y', colors=p2.get_color(),**tkw)
ax.tick_params(axis='x')
ax.legend(handles=[p1, p2])
plt.show() # show the correlation plot of sentiment score*popularity and price trend