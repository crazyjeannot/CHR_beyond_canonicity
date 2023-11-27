import pickle
import pandas as pd
import spacy
from spacy import displacy
import csv
import os
from spacy.tokens import Doc
import argparse
from pathlib import Path
from tqdm import tqdm


spacy.prefer_gpu()
SPACY_PIPE = spacy.load('fr_dep_news_trf')

SPACY_PIPE.max_length = 2000000
#SPACY_PIPE.disable_pipes('ner')

def sentpos(token):
    """returns the position of a given token in the sentence"""
    return [position for position, tok in enumerate(token.sent) if tok == token][0]


def writesent_pers(doc, output_stream, main_per_column):
    nth = 0
    for position, sentence in enumerate(doc):
        for i, token in enumerate(sentence):
            if nth >= len(main_per_column):
                main_per_column.append("")  # Add an empty string if main_per_column is shorter
            print(i, token, token.lemma_, token.pos_, sentpos(token.head), token.dep_, main_per_column[nth], sep="\t",
                  file=output_stream)
            nth += 1


def output_booknlp_table(file_sentences, file_per_position, output_file):
    """
    file_sentences: text file with sentences, created when running "tsv_2_sentences_and_character"
    file_per_position: csv file, column 1 = tokens, column 2 = character id if token is a character.
        created when running "tsv_2_sentences_and_character"
    output_file: .bnlp file
    """
    docs = []
    with open(file_sentences) as file:
        sentences = file.readlines()
        for sentence in sentences:
            doc = Doc(SPACY_PIPE.vocab, sentence.split())
            docs.append(SPACY_PIPE(doc))

    with open(file_per_position, newline='') as csvfile:
        pos_perso = csv.reader(csvfile, quoting=csv.QUOTE_NONE)
        info_booknlp = [row[1] for row in pos_perso if row[1] != 'booknlp']

    with open(output_file / ('output_parser_' + str(file_sentences.stem)), "w", encoding="utf8") as output_stream:
        writesent_pers(docs, output_stream, info_booknlp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--input_dir_sentences', help='directory with sentences files, created when running "tsv_2_sentences_and_character', required=True)
    parser.add_argument(
        '-i', '--input_dir_PER_position', help='directory with csv files, column 1 = tokens, column 2 = character id if token is a character. created when running "tsv_2_sentences_and_character"', required=True
    )
    parser.add_argument(
        '-o', '--output', help='output directory, 1 file per book', required=True)
    args = vars(parser.parse_args())

    input_sentences = Path(args["input_dir_sentences"])
    input_dir_csv = Path(args["input_dir_PER_position"])
    outputDir = Path(args["output"])

    # sentences and csv must be in the same order in the 2 directories
    for sentences_file, csv_file in tqdm(zip(input_sentences.iterdir(), input_dir_csv.iterdir()),
                                         total=len(os.listdir(input_dir_csv))):
        output_booknlp_table(sentences_file, csv_file, outputDir)
