import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import numpy as np
import gensim
import pickle

def get_doc2vec_list(list_token, model):
    doc2vec_vec_list = []

    for novel in tqdm(list_token):
        inferred_vector = model.infer_vector(novel)
        doc2vec_vec_list.append(inferred_vector)

    return doc2vec_vec_list


def prepare_training(list_novels, tokens_only=False):
    if tokens_only:
        for i in range(2900, 2960):
            tokens = list_novels[i]
            yield tokens
    else:# For training data, add tags
        for i in range(2900):
            tokens = list_novels[i]
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def train_doc2vec(main_list_lemma, model):
    train_corpus = list(prepare_training(main_list_lemma))
    test_corpus = list(prepare_training(main_list_lemma, tokens_only=True))
    print("BUILD VOCAB")
    model.build_vocab(train_corpus)
    print("TRAIN MODEL")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    return model


if __name__ == '__main__':

    # LOAD DATA
    print("LOAD DATA")

    # SPACY
    index_spacy = pickle.load(open('lemmatized_lists/main_list_index_spacy.pkl', 'rb'))
    lemma_spacy = pickle.load(open('lemmatized_lists/main_list_lemma_spacy.pkl', 'rb'))
    #token_spacy = pickle.load(open('lemmatized_lists/main_list_token_spacy.pkl', 'rb'))
    #pos_spacy = pickle.load(open('lemmatized_lists/main_list_pos_spacy.pkl', 'rb'))


    # STANZA
    index_stanza = pickle.load(open('lemmatized_lists/main_list_index_stanza.pkl', 'rb'))
    lemma_stanza = pickle.load(open('lemmatized_lists/main_list_lemma_stanza.pkl', 'rb'))
    #token_stanza = pickle.load(open('lemmatized_lists/main_list_token_stanza.pkl', 'rb'))
    #pos_stanza = pickle.load(open('lemmatized_lists/main_list_pos_stanza.pkl', 'rb'))

    # LOAD PRE TRAINED MODEL
    # print("LOAD PRE TRAINED DOC2VEC MODEL")
    # model = pickle.load(open('model.pkl','rb'))

    print("TRAIN DOC2VEC MODELS")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=10, epochs=40)
    model_spacy = train_doc2vec(lemma_spacy, model)
    model_stanza = train_doc2vec(lemma_stanza, model)

    print("GET SPACY VECTORS")
    doc2vec_vec_list_spacy = get_doc2vec_list(lemma_spacy, model_spacy)

    print("GET STANZA VECTORS")
    doc2vec_vec_list_stanza = get_doc2vec_list(lemma_stanza, model_stanza)

    print("SAVE MODELS")
    with open('data_doc2vec/chapitres_spacy_model.pkl', 'wb') as f1, open('data_doc2vec/chapitres_stanza_model.pkl', 'wb') as f2:
            pickle.dump(model_spacy, f1)
            pickle.dump(model_stanza, f2)

    print("SAVE DATA DF")
    columns = [i for i in range(0,2000)]
    # SPACY
    df_doc2vec_all_corpora_spacy = pd.DataFrame(doc2vec_vec_list_spacy, columns=columns)
    df_doc2vec_all_corpora_spacy.index = index_spacy
    df_doc2vec_all_corpora_spacy.to_csv('data_doc2vec/spacy_vectors.csv', index=True)

    # STANZA
    df_doc2vec_all_corpora_stanza = pd.DataFrame(doc2vec_vec_list_stanza, columns=columns)
    df_doc2vec_all_corpora_stanza.index = index_stanza
    df_doc2vec_all_corpora_stanza.to_csv('data_doc2vec/stanza_vectors.csv', index=True)
