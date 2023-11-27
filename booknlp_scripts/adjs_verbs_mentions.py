import pickle
import argparse
from tqdm import tqdm
from glob import glob
import os
from pathlib import Path
from collections import Counter

def tree_from_heads(conll_sentence):  # conll: parsed sentence
    children = [[] for _ in range(len(conll_sentence))] # children: list with at index i, the list corresponding to the indices of tokens pointing towards token i
    for idx, line in enumerate(conll_sentence):
        head = line[4]
        children[int(head)].append(idx)
    return children


def read_file(path):
    with open(path) as file:
        lines = file.read()
        all_sentences = []
        doc = []
        sentence = []
        for line in lines.split("\n")[:-1]:
            line = line.split("\t")
            if line[0] == '0' and sentence != []:
                doc.append(tree_from_heads(sentence))
                all_sentences.append(sentence)
                sentence = []
            sentence.append(line)
    return all_sentences, doc

def get_related_adj_and_verbs(perso, path):
    """
    perso: integer corresponding to a character (last column of the conll table)
    path: path of the conll file
    returns a list of the adjectives associated to this character, and another for the verbs
    """
    mentions = []
    related_adj = []
    related_verbs = []
    sentences, dep_indices = read_file(path)
    for sentence, id_in_sentence in zip(sentences, dep_indices):
        for line in sentence:
            if line[6] == str(perso):
                mentions.append(line[1].lower())
                for i, id_list in enumerate(id_in_sentence):
                    if int(line[0]) in id_list: # doublons possibles quand 2 mots réfèrent au même perso dans la même phrase
                        if sentence[i][3] == 'ADJ':
                            if related_adj == [] or sentence[i][1] != related_adj[-1]:  # si la liste est encore vide ou si l'adjectif a été ajouté l'itération d'avant (il y a des chances que ce qoit une autre mention du même perso dans la même phrase)
                                related_adj.append(sentence[i][1].lower())
                        elif sentence[i][3] == 'VERB':
                            if related_verbs == [] or sentence[i][1] != related_verbs[-1]:
                                related_verbs.append(sentence[i][1].lower())  # add the verb related to the character
    return related_adj, related_verbs, mentions


def get_output_file(input_dir, output_file, n):
    """
    input_file: path of the conll file (obtained when running "spacy_dep_parser.py")
    output_file: path for output. tsv file with: perso, id of perso, token, lemma, POS (adj/verb), dep of perso/token
    n: number of characters.
    """
    with open(output_file, "w") as output_stream:
        print("Titre du livre", "Identifiant du personnage", "Annotation manuelle du genre", "Mentions", "Adjectifs", "Verbes", sep="\t", file=output_stream)
        for filename in tqdm(input_dir.iterdir(), total=len(os.listdir(input_dir))):
            for id_per in range(1, n+1):
                adjs, verbs, mentions = get_related_adj_and_verbs(id_per, filename)
                #adjs = [adj[0] for adj in Counter(adjs).most_common(min(len(adjs), 20))]
                #verbs = [verb[0] for verb in Counter(verbs).most_common(min(len(verbs), 20))]
                #mentions = [mention[0] for mention in Counter(mentions).most_common(min(len(mentions), 20))]
                print(filename.stem[14:], id_per, '', ', '.join(mentions), ', '.join(adjs), ', '.join(verbs), sep="\t", file=output_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', help='path of the input directory containing files (obtained when running "spacy_dep_parser.py")'
    )
    parser.add_argument(
        '-o', '--output_file', help='path for output file. 1 line per character. columns: list of adjs, list of verbs, mentions of the character'
    )
    parser.add_argument(
        '-n', '--nb_characters', help='the n main characters to take into account for each book', default=1, type=int, required=True
    )

    args = vars(parser.parse_args())
    inputdir = Path(args["input_dir"])
    outputDir = args["output_file"]
    n = args['nb_characters']

    get_output_file(inputdir, outputDir, n)
