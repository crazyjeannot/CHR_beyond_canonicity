import pandas as pd
import numpy as np
import pickle
import joblib
import random
from bertopic import BERTopic
from nltk.corpus import stopwords
import argparse
from umap import UMAP

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

seed_topic_list = [
    ['survie', 'danger', 'résilience', 'nature', 'courage', 'ruse', 'force', 'isolement', 'ingéniosité', 'ressources', 'persévérance'],
    ['révolte', 'rébellion', 'injustice', 'résistance', 'combat', 'oppression', 'liberté', 'courage', 'révolution', 'réforme', 'insurrection', 'conspiration'],
    ['aventure', 'maritime', 'navire', 'océan', 'voyage', 'tempête', 'capitaine', 'piraterie', 'exploration', 'naufrage', 'mer', 'exotisme'],
    ['conflits', 'guerres', 'bataille', 'stratégie', 'courage', 'héroïsme', 'trahison', 'danger', 'armes', 'victoire', 'soldats', 'sacrifice'],
    ['nature', 'sauvage', 'exploration', 'survie', 'beauté', 'danger', 'faune', 'flore', 'environnement', 'aventure', 'paysage'],
    ['amitié', 'loyauté', 'compagnons', 'entraide', 'partage', 'fraternité', 'solidarité', 'conflit', 'dévouement', 'camaraderie', 'alliance'],
    ['royauté', 'reine', 'roi', 'couronne', 'palais', 'monarchie', 'noblesse', 'dynastie', 'trône', 'pouvoir', 'succession'],
    ['révolution', 'française', 'liberté', 'égalité', 'fraternité', 'guillotine', 'révolution', 'terreur', 'bastille', 'république', 'louis', 'xvi', 'robespierre'],
    ['empire', 'colonial', 'exploration', 'exploitation', 'exotisme', 'domination', 'résistance', 'culture', 'civilisation', 'conflit', 'commerce', 'découverte'],
    ['classes', 'populaires', 'travail', 'pauvreté', 'injustice', 'lutte', 'quotidien', 'famille', 'survie', 'solidarité', 'exploitation', 'espoir'],
    ['religion', 'spiritualité', 'foi', 'église', 'hérésie', 'spiritualité', 'conflit', 'croyances', 'rituels', 'monastère', 'péché', 'rédemption'],
    ['crime', 'meurtre', 'violence', 'mystère', 'danger', 'victime', 'suspect', 'mobile', 'preuve', 'conséquence', 'loi'],
    ['détective', 'investigation', 'logique', 'résolution', 'indices', 'mystère', 'intuition', 'observateur', 'perspicacité', 'audace', 'déduction'],
    ['suspense', 'tension', 'mystère', 'attente', 'surprise', 'angoisse', 'rebondissement', 'danger', 'anticipation', 'incertitude', 'peur'],
    ['corruption', 'pouvoir', 'trahison', 'manipulation', 'argent', 'immoralité', 'crime', 'justice', 'secret', 'influence', 'avidité'],
    ['conspiration', 'secret', 'manipulation', 'pouvoir', 'trahison', 'mystère', 'complots', 'paranoïa', 'danger', 'révélation', 'contrôle'],
    ['mystère', 'énigme', 'secret', 'suspense', 'découverte', 'curiosité', 'surprise', 'intrigue', 'révélation', 'tension', 'peur'],
    ['policier', 'psychologique', 'psychologie', 'crime', 'suspense', 'caractère', 'motivation', 'manipulation', 'mystère', 'peur', 'introspection', 'tension'],
    ['famille', 'frères', 'et', 'sœurs', 'parents', 'amour', 'soutien', 'conflit', 'liens', 'tradition', 'maison', 'génération', 'héritage'],
    ['drame', 'conflit', 'dispute', 'trahison', 'séparation', 'désaccord', 'rupture', 'tension', 'jalousie', 'malentendu', 'réconciliation', 'déception'],
    ['réconciliation', 'résolution', 'pardon', 'compromis', 'dialogue', 'amour', 'retrouvé', 'paix', 'résolution', 'réunion', 'réconciliation', 'acceptation', 'réparation'],
    ['mariage',  'engagement', 'amour', 'vœux', 'union', 'engagement', 'célébration', 'promesse', 'bonheur', 'cérémonie', 'alliance', 'famille'],
    ['triangle', 'amoureux', 'choix', 'jalousie', 'désir', 'conflit', 'passion', 'indecision', 'relation', 'confusion', 'rivalité', 'désir'],
    ['séduction',  'flirt', 'attirance', 'charme', 'jeu', 'désir', 'complicité', 'romantisme', 'flirt', 'passion', 'intrigue', 'tentation'],
    ['famille', 'relations', 'traditions', 'soutien', 'conflits', 'amour', 'génération', 'héritage', 'parents', 'fratrie', 'liens', 'familiaux'],
    ['éducation', 'apprentissage', 'école', 'université', 'professeurs', 'apprentissage', 'discipline', 'études', 'développement', 'connaissance', 'maturité', 'carrière'],
    ['carrière', 'travail', 'ambitions', 'réussites', 'échecs', 'profession', 'collègues', 'défis', 'compétences', 'métier', 'succès', 'détermination'],
    ['politique', 'société', 'convictions', 'activismes', 'mouvements', 'sociétés', 'changements', 'événements', 'participation', 'histoire', 'influence', 'opinion'],
    ['voyage', 'découverte', 'pays', 'cultures', 'aventures', 'explorations', 'rencontres', 'exotisme', 'dépaysement', 'appréciation', 'découverte', 'apprentissage'],
    ['magie', 'sorcellerie', 'sorts', 'rituels', 'potions', 'prophéties', 'enchantements', 'mystique', 'sorcières', 'grimoires', 'transformation', 'surnaturel'],
    ['épouvante', 'horreur', 'peur', 'suspense', 'terreur', 'frisson', 'cauchemar', 'mort', 'obscurité', 'malédiction', 'mystère', 'monstres'],
    ['légendes', 'folklore', 'mythes', 'contes', 'traditions', 'superstitions', 'créatures', 'mythiques', 'récits', 'ancestraux', 'mystères', 'fables', 'folklore', 'légendes'],
    ['fantastique', 'urbain', 'ville', 'modernité', 'magie', 'créatures', 'surnaturelles', 'occulte', 'mystère', 'conflit', 'humain', 'réalité', 'secret'],
    ['paranormal', 'psychique', 'médium', 'esprits', 'télépathie', 'hantise', 'prémonition', 'réincarnation', 'clairvoyance', 'fantômes', 'possession', 'aura'],
    ['romance',  'amour', 'relation', 'affection', 'amour', 'dévotion', 'intimité', 'tendresse', 'connexion', 'sensualité', 'attachement', 'amour'],
    ['plaisir',  'extase', 'sensation', 'orgasme', 'toucher', 'corps', 'jouissance', 'sensualité', 'passion', 'désir', 'libération', 'intimité'],
    ['tabou', 'interdit', 'secret', 'désir', 'risque', 'transgression', 'érotisme', 'passion', 'danger', 'tentation', 'limite', 'exploration'],
    ['exploration',  'découverte', 'curiosité', 'corps', 'sensualité', 'toucher', 'désir', 'intimité', 'jouissance', 'érotisme', 'passion', 'connaissance'],
    ['intimité','vulnérabilité', 'confiance', 'émotion', 'partage', 'tendresse', 'corps', 'amour', 'dévotion', 'désir', 'intimité', 'sensualité'],
    ['corps','beauté', 'désir', 'sensualité', 'admiration', 'attraction', 'érotisme', 'toucher', 'regard', 'corps', 'beauté', 'passion']]


def concatenate_sentences(list_of_lists, N):
    concatenated_lists = []

    for inner_list in list_of_lists:
        concatenated_inner_list = []
        num_sentences = len(inner_list)

        for i in range(0, num_sentences, N):
            concatenated_sentence = ' '.join(inner_list[i:i+N])
            concatenated_inner_list.append(concatenated_sentence)
        concatenated_lists.append(concatenated_inner_list)

    return concatenated_lists

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


def rescale(x, inplace=False):
    """ Rescale an embedding so optimization will not have convergence issues.
    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x

if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--i_sentences_per_chunk', help='select i sentences per chunk', default=5, type=int
    )
    parser.add_argument(
        '-j', '--j_random_chunks', help='select j random chunks in novel', default=300, type=int
    )

    args = vars(parser.parse_args())

    I = args['i_sentences_per_chunk']
    J = args['j_random_chunks']

    print("LOAD LEMMAS SENTENCES")
    chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences.pkl')
    chapitres_tokens_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_tokens_stanza_sentences.pkl')
    chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences.pkl')

    print("SELECT SENTENCES")
    concatenated_lists = concatenate_sentences(chapitres_tokens_sentences, I)
    selected_sentences, selected_classes = select_random_sentences(concatenated_lists, chapitres_index_sentences, J)

    all_sentences, all_classes = select_all_sentences(concatenated_lists, chapitres_index_sentences)

    print("TOPIC MODEL PARAMETERS")
    cluster_model = KMeans(n_clusters=100)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    print("CREATE EMBEDDINGS")
    #sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
    #embeddings = sentence_model.encode(selected_sentences, show_progress_bar=True)
    #with open('embeddings.pkl', 'wb') as f1:
    #    pickle.dump(embeddings, f1)
    embeddings = joblib.load('embeddings.pkl')


    #print("GET EMBEDDINGS ALL CORPORA")
    #novel_vectors = []

    #for novel in all_sentences:
    #    paragraph_vectors = model.encode(novel, show_progress_bar=True)
        ## paragraph_vectors.shape = (N_sentences, D)
    #    novel_vector = np.mean(paragraph_vectors, axis=0)
        ## sentences_vectors.shape = (1,D)
    #    novel_vectors.append(novel_vector)

    #with open("EMBEDDINGS_CHAPITRES.npy", 'wb') as f:
    #    np.save(f, np.array(novel_vectors))


    print("DIMENSIONALITY REDUCTION")
    # Initialize and rescale PCA embeddings
    pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))
    # Start UMAP from PCA embeddings
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        init=pca_embeddings,
    )

    print("TRAIN TOPIC MODEL")
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=cluster_model, ctfidf_model=ctfidf_model, seed_topic_list=seed_topic_list, top_n_words=5, min_topic_size=10, nr_topics ='100', verbose=True)
    topics, probs = topic_model.fit_transform(selected_sentences, embeddings)

    #print("FINE TUNE TOPIC REPRESENTATION")
    #representation_model = KeyBERTInspired()
    #sw_french = stopwords.words('french')
    #vectorizer_model = CountVectorizer(stop_words=sw_french)
    #topic_model.update_topics(selected_sentences, vectorizer_model=vectorizer_model)


    print("GET TOPICS PER CLASS IN ALL CORPORA")
    topics_per_class = topic_model.topics_per_class(selected_sentences, classes=selected_classes)
    fig = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=20)
    fig.write_html("topics_per_class_"+str(J)+"_guided.html")

    print("SAVE DF TO CSV ZIP")
    compression_opts = dict(method='zip', archive_name='topics_per_class_chapitres_'+str(J)+'_guided.csv')
    topics_per_class.to_csv('topics_per_class_chapitres_'+str(J)+'_guided.zip', header=True,  index=False, compression=compression_opts)

    print("SAVE MODEL")
    topic_model.save("bertopic_chapitres_"+str(J)+"_guided.pkl", serialization="pickle")
