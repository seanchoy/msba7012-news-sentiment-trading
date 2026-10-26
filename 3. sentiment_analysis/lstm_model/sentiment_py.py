import numpy
#coding=utf-8

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
import numpy as np
import keras
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import jieba as jb
import re
import keras.backend as backend
from keras.regularizers import l2



# define function
# remove_punctuation
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist(r"stopwords.txt")



# read the data
df = pd.read_csv(r'train3.csv')
df = df.sample(frac=1.0).reset_index(drop=True)

# remove punctuation
df['clean_review'] = df['title'].apply(remove_punctuation)
# remove stop word
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))

# Build LSTM

## data preprocessing

MAX_NB_WORDS = 33684
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 60
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cut_review'].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df['cut_review'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
Y = pd.get_dummies(df['label']).values #多类标签的onehot展开

# train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# fit the model
epochs = 2
batch_size = 64

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(60, dropout=0.3, recurrent_dropout=0.5)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model_lstm = model.fit(X_train, Y_train, epochs= epochs, batch_size=batch_size,validation_split=0.1,)
model.reset_states()


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
    return cat_id


# predict
for year in [2017,2018,2019,2020,2021]:
    df_test = pd.read_excel(r'data_{}_for_sc.xlsx'.format(year)) # read the data
    df_test['clean_review'] = df_test['Title'].apply(remove_punctuation)
    df_test['cut_review'] = df_test['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
    label_list = []
    for i in range(df_test.shape[0]):
        title = df_test.iloc[i,-1]
        label_list.append(predict(title))
        backend.clear_session()
    df_test['pred_label']  = label_list
    df_test.to_csv('../res_lstm_{}.csv'.format(year), index = None)
    print('{} is finished'.format(year))