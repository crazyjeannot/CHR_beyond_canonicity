import pandas as pd
import numpy as np
import argparse
import os
import csv
#import joblib
from glob import glob
from tqdm import tqdm
from collections import Counter
from operator import add
import pickle

def get_index(df, n, output_per2id):
    """
    n: number of characters
    """
    per2index = {} #keys: characters, values: list of indexes where the character is
    per2id = {} #keys: characters, values: id (int from 1 to n) of each character
    list_characters = [el[0] for liste in [(Counter(list(df['coref'].dropna())).most_common(n))] for el in liste]
    for id, per in enumerate(list_characters, 1):
        per2index[per] = []
        per2id[per] = id
    for index, row in df.iterrows():
        if row['coref'] in list_characters:
            per2index[row['coref']].append(index)
    print(per2id)
    with open(output_per2id, "wb") as file:
        pickle.dump(per2id, file)
    return per2index, per2id

def get_tokens_before_PER(df, index):
    tokens_before = []
    punct = ['.', '!', ':']
    while df.iloc[index-1][0] not in punct:
        index-=1
        tokens_before.append(df.iloc[index][0])
    return list(tokens_before)

def get_tokens_after_PER(df, index):
    tokens_after = []
    punct = ['.', '!', ':']
    while df.iloc[index][0] not in punct and index+1 < len(df):
        index+=1
        tokens_after.append(df.iloc[index][0])
    return list(tokens_after)


def get_PER_sentences(df, n, output_per2id):
    """
    n: number of characters to take into account.
    """
    list_sentences_PER, list_sentences_BookNLP = [], []
    dict_index, dict_id = get_index(df, n, output_per2id)
    index_all = []
    for id in dict_index.values():
        index_all += id

    for index in index_all:
        per = df.iloc[index][0]
        tokens_before = get_tokens_before_PER(df, index)
        tokens_after = get_tokens_after_PER(df, index)

        tokens_before = list(tokens_before[::-1])
        for character in dict_index:
            if index in dict_index[character]:
                i = dict_id[character]
        sentence_PER = tokens_before+[per]+tokens_after
        book_nlp_PER = [0 for elem in range(len(tokens_before))]+[i]+[0 for elem in range(len(tokens_after))]

        list_sentences_PER.append(sentence_PER)
        list_sentences_BookNLP.append(book_nlp_PER)
    return list_sentences_PER, list_sentences_BookNLP

def get_tokens(tokens):
    flattened_tokens = [token for sublist in tokens for token in sublist]
    return flattened_tokens


def merge_lists(sentences, book_nlp):
    """
    Merge the two input lists based on certain conditions.

    Args:
    - sentences (list of tuples): A list of sentences, where each sentence is a tuple of tokens.
    - book_nlp (list of lists of tuples): A list of BookNLP outputs for each sentence.

    Returns:
    - A tuple of two lists: The merged list of sentences and the merged list of BookNLP outputs.
    """
    merged_sentences, merged_booknlp = [], []

    for i in range(len(sentences)):
        sentence = sentences[i]
        booknlp = book_nlp[i] if i < len(book_nlp) else []

        if sentence in merged_sentences:
            index = merged_sentences.index(sentence)
            merged_booknlp[index] = list(map(add, merged_booknlp[index], booknlp))
        else:
            merged_sentences.append(sentence)
            merged_booknlp.append(booknlp)

    return merged_sentences, merged_booknlp




if __name__ == '__main__':
    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing tsv files', required=True
    )
    parser.add_argument(
        '-o', '--output', help='output directory to write csv of main PER', required=True
    )
    parser.add_argument(
        '-s', '--output_sentences', help='output directory to write sentences of main PER', required=True
    )
    parser.add_argument(
        '-n', '--nb_characters', help='the n main characters to take into account for each book', default=1, type=int
    )

    parser.add_argument(
        '-p', '--file_per2id', help='output file with the per2id dict (from each of the n characters to their id)', default="per2id.pkl"
    )


    args = vars(parser.parse_args())
    inputDir = args["input"]
    outputDir = args["output"]
    outputDir_sentences = args["output_sentences"]
    n = args['nb_characters']
    output_per2id = args['file_per2id']
    #créer le dossier de sortie si inexistant
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    #parcourir les fichiers tsv un par un
    for filename in tqdm(glob(os.path.join(inputDir, '*.tsv'))):
        doc_name = os.path.splitext(os.path.basename(filename))[0]
        print(doc_name)
        try:
            df_Preds = pd.read_csv(filename, sep = '\t', names=['tokens', 'entity', 'coref'], quoting=csv.QUOTE_NONE)

            list_sentences_PER, list_preds_BookNLP = get_PER_sentences(df_Preds, n, output_per2id)

            sentences_PER, booknlp_PER = merge_lists(list_sentences_PER, list_preds_BookNLP)

            df_temp = pd.DataFrame(zip(get_tokens(sentences_PER), get_tokens(booknlp_PER)), columns=['tokens', 'booknlp'])

            df_temp.to_csv(outputDir+doc_name+'_PER_sentences.csv', index=False)
        except pd.errors.ParserError as e:
            print(f"Error reading file: {filename}")
            print(e)
            continue

        # to get all the sentences where the n main characters appear:
        with open(outputDir_sentences+"sentences_"+doc_name, "w", encoding="utf8") as output_stream:
            for sentence in sentences_PER:
                sentence = " ".join(sentence)#str ? weird - otherwise catch exception
                print(sentence, sep="\n", file=output_stream)
