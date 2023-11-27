import numpy as np
import pandas as pd
import joblib
import argparse
from tqdm import tqdm
from nltk import ngrams
from collections import Counter

def ngram_list(sentences, n_gram_len):
    # Flatten the list of sentences and split into words
    words = [word for sentence in sentences for word in sentence.split()]
    # Generate n-grams
    n_grams = ngrams(words, n_gram_len)
    return n_grams


def generate_n_grams_dico(n):
    dico_list_ngrams = {}
    for i in range(1, n+1):
        list_name = f"list_{i}_gram"
        dico_list_ngrams[list_name]=[]

    return dico_list_ngrams


def get_selected_ngrams(chapitres_lemmas_sentences, len_ngrams=5, m_most_common=1000):
    DICT_TEMP = generate_n_grams_dico(len_ngrams)#key = "list_{i}_gram" avec list vide   
    DICT_FINAL = {}

    print("PROCESS_NOVELS")
    # Process each novel
    for sentences in tqdm(chapitres_lemmas_sentences):#, total=len(chapitres_lemmas_sentences)):
        # Calculate n-gram frequencies for n=1 to n=len_ngrams
        for i in range(1, len_ngrams+1):
            DICT_TEMP[f"list_{i}_gram"].extend(ngram_list(sentences, i))# too big for memory ?

    print("DICT_MOST_COMMON_NGRAMS_TO_EXTRACT")
    for i in range(1, len_ngrams+1):
        DICT_FINAL[f"list_{i}_gram"] = list(dict(Counter(DICT_TEMP[f"list_{i}_gram"]).most_common(m_most_common)).keys())

    return DICT_FINAL

def ngram_frequencies(sentences, n_gram_len, n_grams_selected):
    # Flatten the list of sentences and split into words
    words = [word for sentence in sentences for word in sentence.split()]
    # Generate n-gram
    n_grams_all = list(ngrams(words, n_gram_len))
    # filter ngrams
    ngrams_filtered = [ngram for ngram in n_grams_all if ngram in n_grams_selected]
    # Count n-grams
    ngram_counts = Counter(ngrams_filtered)
    ngram_counts_total = Counter(n_grams_all)
    # Calculate relative frequencies
    total_ngrams = sum(ngram_counts_total.values())
    return {'_'.join(list(ngram)): count / total_ngrams for ngram, count in dict(ngram_counts).items()}

def moulinette(chapitres_index_sentences, chapitres_lemmas_sentences, dict_features_to_extract, len_ngrams=5):

    data = {}
    
    # Process each novel
    for novel_name, sentences in tqdm(zip(chapitres_index_sentences, chapitres_lemmas_sentences), total=len(chapitres_index_sentences)):
        # Initialize a dictionary for this novel
        novel_data = {}
        # Calculate n-gram frequencies for n=1 to n=len_ngrams
        for i in range(1, len_ngrams+1):
            novel_data.update(ngram_frequencies(sentences, i, dict_features_to_extract[f"list_{i}_gram"]))
        # Add the novel data to the main dictionary
        data[novel_name] = novel_data

    # Create the DataFrame
    df_ngram_freq = pd.DataFrame.from_dict(data, orient='index')

    # Replace NaN with 0 (for n-grams that do not appear in some novels)
    #df_ngram_freq.fillna(0, inplace=True)

    # Display the DataFrame
    return df_ngram_freq


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--n_len_ngrams', help='N len max ngrams', default=5, type=int
    )
    parser.add_argument(
        '-m', '--m_most_common', help='M most common ngrams to extract in each novels', default=1000, type=int
    )

    args = vars(parser.parse_args())

    N = args['n_len_ngrams']
    M = args['m_most_common']

    print("LOAD CHAPITRES SENTENCES")
    chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences_chapitres.pkl')    
    chapitres_tokens_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_tokens_stanza_sentences_chapitres.pkl')
    chapitres_pos_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_pos_stanza_sentences_chapitres.pkl')
    chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences_chapitres.pkl')

    print("GET NGRAMS TO EXTRACT")
    random_indices = np.random.choice(len(chapitres_lemmas_sentences), size=800, replace=False)
    dict_main = get_selected_ngrams([chapitres_lemmas_sentences[i] for i in random_indices], N, M)
    
    print("GET FREQUENCIES")
    df_res = moulinette(chapitres_index_sentences, chapitres_lemmas_sentences, dict_main, N)

    print("SAVE DATAFRAME")
    df_final.to_csv("AWARDS_STEPS_"+str(N)+"ngrams_"+str(M)+"most_common.csv")