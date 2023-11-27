import pandas as pd
import numpy as np
import argparse
import os
import csv
from glob import glob
from tqdm import tqdm
from collections import Counter
from operator import add
import pickle


def get_chronotope(df):
    loc_list = []
    time_list = []
    fac_list = []

    current_loc = ''
    current_time = ''
    current_fac = ''

    for _, row in df.iterrows():
        token = row['tokens']
        entity = row['entity']

        if entity.startswith('B-LOC'):
            if current_loc:
                loc_list.append(current_loc)
            current_loc = token
        elif entity.startswith('I-LOC'):
            current_loc += ' ' + token
        elif entity.startswith('B-TIME'):
            if current_time:
                time_list.append(current_time)
            current_time = token
        elif entity.startswith('I-TIME'):
            current_time += ' ' + token
        elif entity.startswith('B-FAC'):
            if current_fac:
                fac_list.append(current_fac)
            current_fac = token
        elif entity.startswith('I-FAC'):
            current_fac += ' ' + token
        elif entity == 'O':
            if current_loc:
                loc_list.append(current_loc)
                current_loc = ''
            if current_time:
                time_list.append(current_time)
                current_time = ''
            if current_fac:
                fac_list.append(current_fac)
                current_fac = ''
        elif entity.startswith('E-LOC'):
            current_loc += ' ' + token
            loc_list.append(current_loc)
            current_loc = ''
        elif entity.startswith('E-TIME'):
            current_time += ' ' + token
            time_list.append(current_time)
            current_time = ''
        elif entity.startswith('E-FAC'):
            current_fac += ' ' + token
            fac_list.append(current_fac)
            current_fac = ''

    return loc_list, time_list, fac_list



def get_vec(list_token, doc2vec_model):
    return doc2vec_model.infer_vector(list_token)


if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='input directory containing tsv files', required=True
    )
    parser.add_argument(
        '-o', '--output', help='output file to write final Dataframe', required=True
    )
    parser.add_argument(
        '-d', '--doc2vec_model', help='path to doc2vec model', required=True
    )

    args = vars(parser.parse_args())
    inputDir = args["input"]
    outputfile = args["output"]
    doc2vec_path = args["doc2vec_model"]

    model = pickle.load(open(doc2vec_path,'rb'))
    GLOBAL_INDEX, GLOBAL_VEC = [], []

    #parcourir les fichiers tsv un par un
    for filename in tqdm(glob(os.path.join(inputDir, '*.tsv'))):
        doc_name = os.path.splitext(os.path.basename(filename))[0]
        GLOBAL_INDEX.append(doc_name)
        print(doc_name)
        try:
            df_Preds = pd.read_csv(filename, sep = '\t', names=['tokens', 'entity', 'coref'], quoting=csv.QUOTE_NONE)

            time_list, loc_list, fac_list = get_chronotope(df_Preds)


            TIME_vec = get_vec(time_list, model)
            LOC_vec = get_vec(loc_list, model)
            FAC_vec = get_vec(fac_list, model)

            CONCAT_vec = TIME_vec+LOC_vec+FAC_vec
            GLOBAL_VEC.append(CONCAT_vec)

        except pd.errors.ParserError as e:
            print(f"Error reading file: {filename}")
            print(e)
            continue

    columns = ["TIME"+str(i) for in in range(len(TIME_vec))]+["LOC"+str(i) for in in range(len(LOC_vec))]+["FAC"+str(i) for in in range(len(FAC_vec))]
    df_FINAL = pd.DataFrame(GLOBAL_VEC, columns=columns, index=GLOBAL_INDEX)
    df_FINAL.to_csv(outputfile, index=True)
