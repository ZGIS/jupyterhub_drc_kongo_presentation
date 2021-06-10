# -*- coding: utf-8 -*-

from collections import defaultdict
import re
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords

def filter_tweets_before_tokenization(preprocessed_words, reg_expression):
    return [re.sub(reg_expression, '', text) for text in preprocessed_words]

def filter_tweets_after_tokenization(preprocessed_words, reg_expression):
    return [[re.sub(reg_expression,'', string) for string in sub_list] for sub_list in preprocessed_words]

def synonym_handling(preprocessed_words, synonyms, new_term):
    synonyms = set(synonyms)
    document = []
    text_wo_synonyms = []
    for j in range(len(preprocessed_words)):
        for z in range(len(preprocessed_words[j])):
            word = preprocessed_words[j][z]
            if word in synonyms:
                document.append(new_term)
            else:
                document.append(word)
        text_wo_synonyms.append(document)
        document = []
    return text_wo_synonyms

def getFrequency(preprocessed_words):
    frequency = defaultdict(int)
    for text in preprocessed_words:
         for token in text:
            frequency[token] += 1
    return frequency

def preprocessTweetText(text):
    #tweets' text as list
    tweets_text = text.tolist()
    #lowercase
    tweets_text=[tweet.lower() for tweet in tweets_text]

    #remove URLs
    remove_url_regex = r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
    tweets_text = filter_tweets_before_tokenization(tweets_text, remove_url_regex)

    #tokenization
    tweets_text=[nltk.word_tokenize(tweet) for tweet in tweets_text]
    #remove special characters
    remove_sc_regex = r'[^A-Za-z ]+'
    tweets_text = filter_tweets_after_tokenization(tweets_text, remove_sc_regex)

    # remove short words
    remove_short_words_regex = r'\W*\b\w{1,3}\b'
    tweets_text = filter_tweets_after_tokenization(tweets_text, remove_short_words_regex)

    # Remove all user names in the tweet text
    user_names_regex = r"@\S+"
    tweets_text = filter_tweets_after_tokenization(tweets_text,user_names_regex)

    #increase keyword frequency by aggregating similar keywords
    # check the order if preprocessing routine! e.g. stemming would effect the performance of synonym handling
    #disaster = 'hurrican'
    #disaster_terms = ['hurricane', 'hurricaneharvey', 'hurricane_harvey', 'flood', 'storm']
    #tweets_text = synonym_handling(tweets_text, disaster, disaster_terms)

    #Remove unique words that appear only once in the dataset
    frequency = getFrequency(tweets_text)
    min_frequency_words = 2
    tweets_text = [[token for token in tweet if frequency[token] > min_frequency_words] for tweet in tweets_text]

    # Remove stop words
    # You need to download the stopwords
    from nltk.corpus import PlaintextCorpusReader
    stoplist = set(stopwords.words('english'))
    tweets_text = [[word for word in document if word not in stoplist] for document in tweets_text]

    #Stemming
    stemmer = PorterStemmer()
    #stemmer = SnowballStemmer("english")
    tweets_text = [[stemmer.stem(word) for word in sub_list] for sub_list in tweets_text]

    #remove empty strings
    tweets_text = [[word for word in document if word] for document in tweets_text]
    tmp= []
    for text in tweets_text:
        tmp.append(' '.join(text))
    tweets_text = tmp
    return tweets_text