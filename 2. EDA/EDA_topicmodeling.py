from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

import jieba.posseg as jp
import random
import sys
import pandas as pd

# text processing
def get_text(texts):
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')
    stopwords = [line.strip() for line in open('cn_stopwords.txt','r', encoding='utf')]  # stop words
    words_list = []
    for text in texts:
        words = [w.word for w in jp.lcut(text) if w.flag in flags and w.word not in stopwords and w.word != 'x000D']
        words_list.append(words)
    return words_list

news_df2017 = pd.read_excel('data_2017_for_sc.xlsx')
news_df2018 = pd.read_excel('data_2018_for_sc.xlsx')
news_df2019 = pd.read_excel('data_2019_for_sc.xlsx')
news_df2020 = pd.read_excel('data_2020_for_sc.xlsx')
news_df2021 = pd.read_excel('data_2021_for_sc.xlsx')
news_df = pd.concat([news_df2017,news_df2018,news_df2019,news_df2020,news_df2021],axis=0)

news_df['Description'] = news_df['Title'].map(str)+' '+news_df['KeyWord'].map(str)+' '+news_df['NewsSummary'].map(str)+' '+news_df['NewsContent'].map(str)


trains = []
tests = []
news = []
for text in news_df['Description']:
    if (sys.getsizeof(text) > 500):
        news.append(text)
# len(news)
text = random.sample(news, 10000)  # randomly get 10000 piece of text
trains.extend(text[0:9000])
tests.append(text[9000:10000])

words_list = get_text(trains)
# print(words_list)

# Dictionary() iterates through all the text, assigning a separate integer ID to each non-repeating word and collecting the number of occurrences of the word and related statistics
dictionary = corpora.Dictionary(words_list)
dictionary.filter_extremes(no_below=200, no_above=0.6)
print(dictionary)

# Converting a dictionary into a bag of words
# doc2bow() converts a dictionary into a bag of words. The result corpus is a list of vectors, and the number of vectors is the number of documents
# Each document vector contains a series of tuples in the form of (word ID, word frequency)
corpus = [dictionary.doc2bow(words) for words in words_list]
print('Output a vector of each document:')
print(corpus)

# LDA topic model
# num_topics -- required, the number of topics to be generated.
# id2word -- required, the LdaModel class requires our previous dictionary to map all ids to strings.
# passes -- optional, the number of times the model traverses the corpus. The more passes, the more accurate the model. But for very large corpora, it can take a long time to traverse too many times.
lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics=20, id2word=dictionary, passes=30)
# print out all words with 50 words each
topic_words = lda_model.print_topics(num_topics=20, num_words=50)
print('print out all words with 50 words each:')
print(topic_words)

# calculate the perplexity of different num of topics
def perplexity(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
    print(2**-(ldamodel.log_perplexity(corpus)))
    return 2**-(ldamodel.log_perplexity(corpus))

# get the plot of perplexity
# The smaller the perplexity the better, but as the number of topics increases, the perplexity decreases slowly, so choose 20 topics
x = range(1,30)
y = [perplexity(i) for i in x]
plt.plot(x, y)
plt.xlabel('topic no.')
plt.ylabel('perplexity')
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.title('topic-perplexity')
plt.show()

# get the model of 20 topics
lda_20topic = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes = 30,random_state=1)
topic_list_20topic=lda_20topic.print_topics()
print(topic_list_20topic)

# output the result with pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
data_20topic = pyLDAvis.gensim.prepare(lda_20topic, corpus, dictionary)
pyLDAvis.save_html(data_20topic, 'topic20.html')

# get the wordcloud of each topic
for t in range(lda_20topic.num_topics):
    frequencies = lda_20topic.show_topic(t, 50)
    frequencies = dict(frequencies)
    plt.figure()
    plt.imshow(WordCloud(font_path='SimHei.ttf',
                         background_color = 'white',
                         max_font_size =150).fit_words(frequencies))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()