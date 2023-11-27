import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import stanza
from os import path
from unicodedata import normalize

path_name = 'corpus_chapitres_txt/*.txt'

STANZA_PIPE = stanza.Pipeline(lang='fr', processors='tokenize, lemma, pos', use_gpu=True)

def clean_text(txt):
    txt_res = normalize("NFKD", txt).replace('\xa0', ' ')
    txt_res = txt_res.replace("\\", "").replace('\\xa0', '')
    return txt_res


def pipeline_stanza_sentences(doc):

    pos_ko = ["NUM", "X", "SYM", "PUNCT"]
    list_sentences_tokens, list_sentences_lemmas, list_sentences_pos, list_sentences_wt_proper_sw = [], [], [], []
    nb_sentences, nb_tokens = 0, 0

    with open(doc, encoding="utf8") as file:
        text = file.readlines()
        text_clean = clean_text(str(text[0]).lower())
        docs = STANZA_PIPE(text_clean)

        nb_sentences += len(docs.sentences)

        for sent in docs.sentences:
            list_tokens, list_lemmas, list_pos, list_wt_proper_sw = [], [], [], []
            nb_tokens += len(sent.words)
            for word in sent.words:
                #si le token est bien un mot on récupère son texte, son lemme et son pos
                if word.upos not in pos_ko:
                    list_tokens.append(word.text)
                    list_lemmas.append(word.lemma)
                    list_pos.append(word.upos)
		# si le token n'est pas un stopwords ou un proper noun on le prend
                if word.upos not in ['DET', 'PRON', 'ADP', 'AUX', 'CONJ', 'SCONJ', 'PART', 'PROPN']+pos_ko:
                    list_wt_proper_sw.append(word.text)
            list_sentences_tokens.append(" ".join(map(str, list_tokens)))
            list_sentences_lemmas.append(" ".join(map(str, list_lemmas)))
            list_sentences_pos.append(" ".join(map(str, list_pos)))
            list_sentences_wt_proper_sw.append(" ".join(map(str, list_wt_proper_sw)))


    return list_sentences_tokens[5:len(list_sentences_tokens)-5], list_sentences_lemmas[5:len(list_sentences_lemmas)-5], list_sentences_pos[5:len(list_sentences_pos)-5], list_sentences_wt_proper_sw[5:len(list_sentences_wt_proper_sw)-5], nb_sentences, nb_tokens

def moulinette(path_name):

    nb_total_tokens, nb_total_sentences = 0, 0
    main_list_token, main_list_lemma, main_list_pos, main_list_wt_proper_sw, main_list_index = [], [], [], [], []

    #print("\n\nBEGIN PROCESSING CORPUS-----------")

    for doc in tqdm(glob(path_name)):

        #print("\n\nBEGIN PROCESSING NOVEL-----------")

        doc_name = path.splitext(path.basename(doc))[0]
        date = doc_name.split("_")[0]
        print(doc_name)

        #On recupere le texte des romans sous forme de listes de lemmes et de pos grâce à stanza

        list_token, list_lemma, list_pos, list_wt_proper_sw, nb_sentences, nb_tokens = pipeline_stanza_sentences(doc)

        nb_total_tokens += nb_tokens
        nb_total_sentences += nb_sentences

        main_list_token.append(list_token)
        main_list_lemma.append(list_lemma)
        main_list_pos.append(list_pos)
        main_list_wt_proper_sw.append(list_wt_proper_sw)
        main_list_index.append(doc_name)

        #print("\n\nEND PROCESSING NOVEL-----------")

    print("NB TOKENS FINAL: ", nb_total_tokens, "NB SENTENCES FINAL: ", nb_total_sentences)
    #print("\n\nEND PROCESSING CORPUS-----------")

    return main_list_lemma, main_list_token, main_list_pos, main_list_wt_proper_sw, main_list_index


if __name__ == '__main__':
    main_list_lemma, main_list_token, main_list_pos, main_list_wt_proper_sw, main_list_index = moulinette(path_name)
    with open('/data/jbarre/lemmatization/LEMMAS_AVENTURES.pkl', 'wb') as f1, open('/data/jbarre/lemmatization/TOKENS_AVENTURES.pkl', 'wb') as f2, open('/data/jbarre/lemmatization/POS_AVENTURES.pkl', 'wb') as f3, open('/data/jbarre/lemmatization/WT_PROPER_SW_AVENTURES.pkl') as f4, open('/data/jbarre/lemmatization/INDEX_AVENTURES.pkl', 'wb') as f5:
            pickle.dump(main_list_lemma, f1)
            pickle.dump(main_list_token, f2)
            pickle.dump(main_list_pos, f3)
            pickle.dump(main_list_wt_proper_sw, f4)
            pickle.dump(main_list_index, f5)
