from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_k(dictionary, doc_term_matrix, doc_clean, stop, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LsiModel(doc_term_matrix, num_topics = num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def process_data(data):
    # Data Preprocessing
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = []
    for i in data:
        raw = str(i).lower()
        # Tokenizing
        tokens = tokenizer.tokenize(raw)
        # Removing stopwords
        stopped_tokens = [j for j in tokens if not j in en_stop]
        # Lemmatizing tokens
        lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens]
        # Remove tokens consisting of one single char
        tokens_without_single = [j for j in lemma_tokens if not len(j) == 1]
        # Remove numeric tokens
        tokens_without_num = [j for j in tokens_without_single if not j.isnumeric()]
        text.append(tokens_without_num)
    print(text)
    return text


def get_matrix(text):
    dict = corpora.Dictionary(text)
    # Filter out tokens which appear in less than 15 docs,
    # which appear in more than 50% of the docs,
    # and then keep only the first 100000 tokens
    dict.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # Generating doc_term_matrix
    doc_term_matrix = [dict.doc2bow(doc) for doc in text]
    #for i in range(len(doc_term_matrix)):
    #    print(doc_term_matrix[i])
    tfidf = models.TfidfModel(doc_term_matrix, smartirs='npu')
    corpus_tfidf = tfidf[doc_term_matrix]
    #for i in range(len(corpus_tfidf)):
    #    print(corpus_tfidf[i])
    return dict, doc_term_matrix, corpus_tfidf


if __name__ == "__main__":
    words = 10
    np.random.seed(2020)
    num_of_topics = 6

    # Load Data
    selected_data = []
    data = pd.read_csv('./voted-kaggle-dataset.csv')
    col_1 = data['Title']
    col_2 = data['Subtitle']
    for i in range(len(col_1)):
        full_row = str(col_1[i]) + " " + str(col_2[i])
        #print(full_row)
        selected_data.append(full_row)
        full_row = ""
    #selected_data = col_1.append(col_2)
    text = process_data(selected_data)
    dict, doc_term_matrix, corpus_tfidf = get_matrix(text)
    '''
    start, stop, step = 2, 20, 1
    model_list, coherence_values = find_k(dict, corpus_tfidf, text, stop, start, step)
    print(coherence_values)
    num_of_topics = start + coherence_values.index(max(coherence_values))
    print(num_of_topics)

    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    '''

    # Apply LSA algorithm
    lsamodel = LsiModel(corpus_tfidf, num_topics = num_of_topics, id2word = dict)
    corpus_lsi = lsamodel[corpus_tfidf]
    for doc, as_text in zip(corpus_lsi, text):
        print(doc, as_text)
    #vectorized_corpus = lsamodel[corpus_tfidf]
    #print(vectorized_corpus)
    #print(lsamodel.show_topics(num_topics = num_of_topics, num_words = words, log = False, formatted = False))
    topic_list = []
    for index, topic in lsamodel.show_topics(num_topics = num_of_topics, num_words = words, formatted = False):
        topic_list.append([w[0] for w in topic])
        print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
    print(topic_list)


    test_topic = "Trump's false crusade rolls on despite devastating Supreme Court rebuke"
    print(test_topic)
    test_data = test_topic.split()
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = []
    for i in test_data:
        raw = str(i).lower()
        # Tokenizing
        tokens = tokenizer.tokenize(raw)
        # Removing stopwords
        stopped_tokens = [j for j in tokens if not j in en_stop]
        # Lemmatizing tokens
        lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens]
        # Remove tokens consisting of one single char
        tokens_without_single = [j for j in lemma_tokens if not len(j) == 1]
        # Remove numeric tokens
        tokens_without_num = [j for j in tokens_without_single if not j.isnumeric()]
        text.append(tokens_without_num)
    #print(text)
    corrected_text = []
    for i in range(len(text)):
        if text[i]:
            #print(text[i][0])
            corrected_text.append(text[i][0])
    test_dict = corpora.Dictionary([corrected_text])
    bow_vector = dict.doc2bow(corrected_text)
    for index, score in sorted(lsamodel[bow_vector], key=lambda tup: -1 * tup[1]):
        print("\nScore: {}\t".format(score))
        print('Topic: {} \t Words: {}'.format(index, topic_list[index]))
