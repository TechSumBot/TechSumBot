import gzip
import re
import sys
import string
import json
from os import listdir
from os.path import isfile, join

import gensim
import numpy as np
import nltk
# from nltk.stem import PorterStemmer

from bias_words import word_list as bias_words_list

# Load Google's pre-trained Word2Vec model.
print('loading Google word2vec vectors...')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
translator = str.maketrans('', '', string.punctuation)


# stemmer = PorterStemmer()


def magnify(a):
    maximum = np.max(a)
    minimum = np.min(a)
    return (a - minimum) / (maximum - minimum)


def normalize(a):
    return a / (a.sum(axis=1, keepdims=True) + 0.000000001)


def cosine_similarity(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_cooccurrence_matrix(matrix, tokens, window=2):
    for i, token in enumerate(tokens):
        if token not in matrix:
            matrix[token] = {}
        for j in range(max(i - window, 0), min(i + window, len(tokens))):
            if i == j:
                continue

            adj_token = tokens[j]
            if adj_token not in matrix[token]:
                matrix[token][adj_token] = 0
            matrix[token][adj_token] += 1


def get_word2vec_embeddings(token):
    return w2v_model[token]


def textrank(tokens, bias_token, cooccurrence_matrix, damping_factor=0.85, cooccurrence_minimum_threshold=3,
             biased=False):
    # mapping tokens to IDs
    token_ids_map = {}
    for i, token in enumerate(tokens):
        token_ids_map[token] = i

    # create text rank matrix
    matrix = np.zeros((len(tokens), len(tokens)))
    for token in tokens:
        for adj_token in cooccurrence_matrix[token]:
            if cooccurrence_matrix[token][adj_token] < cooccurrence_minimum_threshold:
                continue

            if token_ids_map[token] != token_ids_map[adj_token]:
                matrix[token_ids_map[token]][token_ids_map[adj_token]] = cooccurrence_matrix[token][adj_token]

    # normalizing textrank matrix
    matrix = normalize(matrix)

    if biased:
        # calculating bias token's word2vec embeddings
        bias_token_embedding = get_word2vec_embeddings(bias_token)
        bias = np.array([cosine_similarity(get_word2vec_embeddings(token), bias_token_embedding) for token in tokens])
        bias = magnify(bias)

        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias

    else:
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) / len(matrix)

    scaled_matrix = normalize(scaled_matrix)

    v = np.ones((len(matrix), 1)) / len(matrix)

    iterations = 40

    for i in range(iterations):
        v = scaled_matrix.T.dot(v)

    return v, tokens


def clean_and_tokenize(datum):
    datum_cleaned = ' '.join(datum.translate(translator).split())
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(datum_cleaned)
    # tokens_cleaned = [stemmer.stem(token) for token in tokens if token not in stopwords and token != 'rt']
    tokens_cleaned = [token for token in tokens if token not in stopwords and token not in ['rt', 'NUM'] and token in w2v_model]
    return tokens_cleaned


def get_filenames_in_directory(path):
    return [filename for filename in listdir(path) if isfile(join(path, filename))]


def load_json(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


def save_as_json(data, filename):
    with open("{}.json".format(filename), "w") as f:
        json.dump(data, f)


def main():
    input_directory = '../data/'
    load_precomputed = sys.argv[1] if len(sys.argv) > 1 else None

    token_dist = nltk.FreqDist()

    if load_precomputed:
        tokens = load_json('../models/tokens_set.json')
        cooccurrence_matrix = load_json('../models/co-occurrence.json')

    else:
        cooccurrence_matrix = {}

        # for all files in the dataset:
        for filename in get_filenames_in_directory(input_directory):
            if 'csv' not in filename:
                continue
            tokens = []
            print('processing file: {}'.format(filename))
            #  read tweet file
            filepath = input_directory + filename
            with gzip.open(filepath, 'rt') as f:
                raw_tweets = f.readlines()

            #   clean its data-- using Scott's script and removing stop words and things like that
            #   build up on existing co-occurrence matrix
            #   add the tokens to the existing tokens set
            tweets = [re.split(r'\t+', raw_tweet)[5] for raw_tweet in raw_tweets]
            tweets_tokenized = [clean_and_tokenize(tweet) for tweet in tweets]

            for tweet_tokenized in tweets_tokenized:
                calculate_cooccurrence_matrix(cooccurrence_matrix, tweet_tokenized)
                tokens.extend(tweet_tokenized)

            token_dist.update(tokens)

        # chopping down the uncommon tokens and removing uncommon words from the co-occurrence matrix
        print('making things small enough to process in memory...')
        tokens = [token for token, freq in token_dist.most_common(30000)]
        for token in token_dist.keys():
            if token not in tokens:
                del cooccurrence_matrix[token]
                continue

            for adj_token in list(cooccurrence_matrix[token].keys()):
                if adj_token not in tokens:
                    del cooccurrence_matrix[token][adj_token]

        # saving co-occurrence matrix and tokens set
        print('saving the intermediate results...')
        save_as_json(cooccurrence_matrix, '../models/co-occurrence')
        save_as_json(tokens, '../models/tokens_set')


    # run normal textrank once on all data, gather top k (e.g. k = 100) keywords
    print('running normal textrank...')
    ranks, _ = textrank(tokens, None, cooccurrence_matrix)
    save_as_json(dict(sorted(zip(tokens, ranks.tolist()), key=lambda pair: pair[1])), '../results/normal_textrank')

    # for each bias_term:
    for bias_word in bias_words_list:
        bias_word = bias_word.lower()
        if bias_word not in w2v_model:
            print('no embeddings for {}, skipping bias word.'.format(bias_word))
            continue

        #   run biased_textrank and gather top n (e.g. n = 50) results
        print('running biased textrank for {}'.format(bias_word))
        # bias_word = stemmer.stem(bias_word)
        biased_ranks, _ = textrank(tokens, bias_word, cooccurrence_matrix, biased=True)
        save_as_json(dict(sorted(zip(tokens, biased_ranks.tolist()), key=lambda pair: pair[1])),
                     '../results/{}'.format(bias_word))


if __name__ == '__main__':
    main()
