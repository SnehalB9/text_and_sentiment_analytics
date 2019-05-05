#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:27:35 2019

@author: Snehal Bhole
"""

import nltk

#####################################
####
#### Sentence Tokenization - Sentence tokenizer breaks text paragraph into sentences.
####
#####################################

from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""

#nltk.download('punkt')
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#####################################
####
#### Word Tokenization - Word tokenizer breaks text paragraph into words.
####
#####################################

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

#####################################
####
#### Frequency Distribution - 
####
#####################################

from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

fdist.most_common(2)

# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

#####################################
####
####  Stopwords - Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.
####
#####################################

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

#####################################
####
####  Removing Stopwords
####
#####################################

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

##################################################
####
####  Lexicon Normalization - Lexicon normalization considers another type of noise in the text. For example, connection, connected, connecting word reduce to a common word "connect". It reduces derivationally related forms of a word to a common root word.
####
##################################################

#####################################
####
####  Stemming - Stemming is a process of linguistic normalization, which reduces words to their word root word or chops off the derivational affixes.
####
#####################################

# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#####################################
####
####  Lemmatization - Lemmatization reduces words to their base word, which is linguistically correct lemmas.
####
#####################################

#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

#nltk.download()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


#####################################
####
####  POS Tagging - The primary target of Part-of-Speech(POS) tagging is to identify the grammatical group of a 
###  given word. Whether it is a NOUN, PRONOUN, ADJECTIVE, VERB, ADVERBS, etc. based on the context. 
###  POS Tagging looks for relationships within the sentence and assigns a corresponding tag to the word.
####
#####################################

sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)
nltk.pos_tag(tokens)

####################################
####                            ####
####  Sentiment Analysis        ####
####                            ####
####################################

#####################################
####
####  Performing Sentiment Analysis using Text Classification - Identifying category or class of given text such as a blog, book, web page, news articles, and tweets.
####
#####################################
# Import pandas
import pandas as pd

data=pd.read_csv('train.tsv', sep='\t')

data.head()

data.info()

data.Sentiment.value_counts()

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

#####################################
####
####  Feature Generation using Bag of Words - 
####
#####################################

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])

# split train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)

## Model Building and Evaluation

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

# Feature Generation using TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])

# split train and test set (TF-IDF)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)

# Model Building and Evaluation (TF-IDF)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

