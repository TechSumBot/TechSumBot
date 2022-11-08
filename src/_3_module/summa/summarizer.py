from math import log10

from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel,BertForSequenceClassification, BertTokenizer, BertModel,RobertaForSequenceClassification
import pickle
from collections import OrderedDict  
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from lexrank import STOPWORDS, LexRank
import _3_module.summa.summarizer as summarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def _set_graph_edge_weights(graph):

    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():
            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity(sentence_1, sentence_2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def get_simcse_embedding(input):
    sim_tokenizer = AutoTokenizer.from_pretrained("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")
    sim_model = AutoModel.from_pretrained("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")

    inputs = sim_tokenizer(input, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = sim_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def _set_graph_edge_weights_simcse(graph):
    embeddings = get_simcse_embedding(graph.nodes())
    for index_1,sentence_1 in enumerate(graph.nodes()):
        for index_2,sentence_2 in enumerate(graph.nodes()):
            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity_simcse(embeddings[index_1], embeddings[index_2])
                if similarity != 0:
                    graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)

def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)

def _get_similarity_simcse(s1, s2):

    cosine_sim_0_1 = 1 - cosine(s1,s2)
    # print(cosine_sim_0_1);exit()
    # return common_word_count / (log_s1 + log_s2)
    return cosine_sim_0_1


def _get_similarity(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score=True):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return [sentence.text for sentence in extracted_sentences]


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, ratio, words,sent_length):
    sentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.


    if not sent_length:
        return sentences
    if words is None and sent_length is None:
        length = len(sentences) * ratio
        return sentences[:int(length)]
    if sent_length:
        return sentences[:int(sent_length)]

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(sentences, words)


def summarize(text, embedding, algorithm='list', ratio=0.2, words=None, language="english", split=False, scores=False, additional_stopwords=None,sent_length=None ):
    if (not isinstance(text, list)) and (not isinstance(text, str)):
        raise ValueError("Text parameter must be a list/str!")

    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text,embedding, language, additional_stopwords, algorithm)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])

    # generate the embedding for initialization
    if embedding == 'tfidf':
        _set_graph_edge_weights(graph)
    if embedding == 'simcse':
        _set_graph_edge_weights_simcse(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)
    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words,sent_length)
    # Sorts the extracted sentences by apparition order in the original text.
    if sent_length:
        extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split, True)


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph