import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
from pprint import pprint
from gensim.models import CoherenceModel
import re
import matplotlib.pyplot as plt

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    word = WordNetLemmatizer().lemmatize(text, pos='v') #Lemmatize
    word = stemmer.stem(word) #Plural to single
    
    return word
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(lemmatize_stemming(token))
    
    return result

def bagofword(processed_data):
    Appearance_word = gensim.corpora.Dictionary(processed_data) 
    Appearance_word.filter_extremes(no_below=5, no_above=0.5, keep_n=500) #filter the result
    
    bags = [Appearance_word.doc2bow(doc) for doc in processed_data]  #appearance count of each word
    
    return bags,Appearance_word

if __name__ == "__main__":

    #dataset 1    
    np.random.seed(2020)
    nltk.download('wordnet')
    data = pd.read_csv('C:/Users/Vanquish/Desktop/data.csv', error_bad_lines=False) #Import data
    data_title = data[['Title']]    #Read the title
    data_subt = data[['Subtitle']]
    #data_title['merge'] = data_title['Title'] + ' ' + data_subt['Subtitle']
    data_ned = data_title['Title'].append(data_subt['Subtitle'])
    #processed_data = data_title['merge'].fillna('').astype(str).map(preprocess)
    processed_data = data_ned.fillna('').astype(str).map(preprocess)

    bags,Appearance_word = bagofword(processed_data) 

    """
    #Determine the k by its coherence value
    coherence_lda = []
    
    for i in range(12):
        lda_model = gensim.models.LdaMulticore(bags, num_topics= i+2, id2word= Appearance_word,passes=2)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=Appearance_word, coherence='c_v')
        coherence_lda.append(coherence_model_lda.get_coherence())
    
    print('\nCoherence Score: ', coherence_lda)
    plt.plot(range(2，len(coherence_lda)，1),coherence_lda)
    plt.ylabel('Coherence Score')
    plt.xlabel('Number of topics')
    plt.show()
    """
    #Evaluate the relevance of a word in the document
    lda_model1 = gensim.models.LdaMulticore(bags, num_topics= 5, id2word= Appearance_word)
    
    print('Topics:\n')
    
    for idx, topic in lda_model1.print_topics(num_words=10):
        mid_list1 = re.findall('\d*\.?\d+',topic)
        mid_list1 = [float(i) for i in mid_list1]
        final_list1 = []
        sum_num = sum(mid_list1)
        
        for num in mid_list1:
            final_list1.append(round(num/sum_num,3))
        
        for j in range(len(mid_list1)):
            topic = topic.replace(str(round(mid_list1[j],3)),str(final_list1[j]))
        
        print('Topic: {} Word: {}'.format(idx, topic))

    print('Testing:\n')
    print('Original document:\n')
    print(processed_data[0])
    for index, score in sorted(lda_model1[bags[0]], key = lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model1.print_topic(index, 10)))
    
    print('Original document:\n')
    test = 'Trump\'s false crusade rolls on despite devastating Supreme Court rebuke'
    print(test)
    bow_vector = Appearance_word.doc2bow(preprocess(test))
    for index, score in sorted(lda_model1[bow_vector], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model1.print_topic(index, 10)))
    
    print('Original document:\n')
    test = 'A doctor who treated some of Houston’s sickest Covid-19 patients has died'
    print(test)
    bow_vector = Appearance_word.doc2bow(preprocess(test))

    for index, score in sorted(lda_model1[bow_vector], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model1.print_topic(index, 10)))

    
    #dataset2
    np.random.seed(2020)
    nltk.download('wordnet')
    data2 = pd.read_csv('C:/Users/Vanquish/Desktop/data2.csv', error_bad_lines=False) #Import data
    data_title2 = data2[['TITLE']]    #Read the title
    data_subt2 = data2[['ABSTRACT']]
    data_title2['merge'] = data_title2['TITLE'] + ' ' + data_subt2['ABSTRACT']

    processed_data2 = data_title2['merge'].fillna('').astype(str).map(preprocess)

    bags2,Appearance_word2 = bagofword(processed_data2) 
    
    #Determine the k by its coherence value
    """
    coherence_lda = []
    
    for i in range(19):
        lda_model = gensim.models.LdaMulticore(bags, num_topics= i+2, id2word= Appearance_word2,passes=2)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=Appearance_word2, coherence='c_v')
        coherence_lda.append(coherence_model_lda.get_coherence())
    
    print('\nCoherence Score: ', coherence_lda)
    plt.plot(range(len(coherence_lda)),coherence_lda)
    plt.ylabel('Coherence Score')
    plt.xlabel('Number of topics')
    plt.show()
    """
    #Evaluate the relevance of a word in the document
    lda_model2 = gensim.models.LdaMulticore(bags2, num_topics= 12, id2word= Appearance_word2)
    
    print('Topics:\n')
    
    for idx, topic in lda_model2.print_topics(num_words=10):
        mid_list2 = re.findall('\d*\.?\d+',topic)
        mid_list2 = [float(i) for i in mid_list2]
        final_list2 = []
        sum_num = sum(mid_list2)
        
        for num in mid_list2:
            final_list2.append(round(num/sum_num,3))
        
        for j in range(len(mid_list2)):
            topic = topic.replace(str(round(mid_list2[j],3)),str(final_list2[j]))
        
        print('Topic: {} Word: {}'.format(idx, topic))

    print('Testing:\n')
    print('Original document:\n')
    print(processed_data2[0])
    for index, score in sorted(lda_model2[bags2[0]], key = lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model2.print_topic(index, 10)))
    
    print('Original document:\n')
    test = 'Trump\'s false crusade rolls on despite devastating Supreme Court rebuke'
    print(test)
    bow_vector2 = Appearance_word2.doc2bow(preprocess(test))
    for index, score in sorted(lda_model2[bow_vector2], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model2.print_topic(index, 10)))
    
    print('Original document:\n')
    test = 'A doctor who treated some of Houston’s sickest Covid-19 patients has died'
    print(test)
    bow_vector2 = Appearance_word2.doc2bow(preprocess(test))

    for index, score in sorted(lda_model2[bow_vector2], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model2.print_topic(index, 10)))
    