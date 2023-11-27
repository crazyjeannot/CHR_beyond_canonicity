import pandas as pd
import numpy as np
import pickle
import joblib
import random
from bertopic import BERTopic
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance

from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP


def select_random_sentences(list_of_lists, list_index, N):
    selected_strings = []
    classes = []
    for inner_list, index in zip(list_of_lists, list_index):
        selected = random.sample(inner_list, min(N, len(inner_list)))
        selected_strings.extend(selected)
        classes.extend([index] * len(selected))
    return selected_strings, classes


def select_all_sentences(list_of_lists, list_index):
    selected_strings = []
    classes = []
    for inner_list, index in zip(list_of_lists, list_index):
        selected_strings.extend(inner_list)
        classes.extend([index] * len(selected_strings))
    return selected_strings, classes



if __name__ == '__main__':

    print("LOAD LEMMAS SENTENCES")
    chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences.pkl')
    chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences.pkl')
    sw_french = stopwords.words('french')

    print("LOAD SENTENCES")
    selected_sentences, selected_classes = select_random_sentences(chapitres_lemmas_sentences, chapitres_index_sentences, 1000)
    #all_sentences, all_classes = select_all_sentences(chapitres_lemmas_sentences, chapitres_index_sentences)

    print("CREATE EMBEDDINGS")
    sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
    embeddings = sentence_model.encode(selected_sentences, show_progress_bar=True)

    print("TOPIC MODEL PARAMETERS")
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    vectorizer_model = CountVectorizer(stop_words=sw_french, strip_accents='unicode')


    print("CREATE TOPIC MODEL")
    topic_model = BERTopic(top_n_words=20, min_topic_size=20, nr_topics = None, language="multilingual", vectorizer_model=vectorizer_model, verbose=True, representation_model=representation_model, umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(selected_sentences, embeddings)

    print("GET TOPICS PER CLASS")
    topics_per_class = topic_model.topics_per_class(selected_sentences, classes=selected_classes)# might have a bug here since we train on selected train sentence -> TO TEST

    print("SAVE DF TO CSV ZIP")
    compression_opts = dict(method='zip', archive_name='topics_per_class_chapitres.csv')
    topics_per_class.to_csv('topics_per_class_chapitres.zip', header=True,  index=False, compression=compression_opts)

    print("SAVE MODEL")
    topic_model.save("bertopic_chapitres", serialization="pickle")
