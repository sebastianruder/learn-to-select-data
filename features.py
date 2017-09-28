"""
Methods for generating features that can be used for Bayesian Optimization.
"""
import numpy as np

import similarity
from constants import SIMILARITY_FUNCTIONS, DIVERSITY_FEATURES


def get_feature_representations(feature_names, examples, trg_examples, vocab,
                                word2vec=None, topic_vectorizer=None,
                                lda_model=None, lowercase=True):
    """
    Retrieve the feature representations of a list of examples.
    :param feature_names: a list containing the names of features to be used
    :param examples: a list of tokenized documents of all source domains
    :param trg_examples: a list of tokenized documents of the target domain
    :param vocab: the Vocabulary object
    :param word2vec: a mapping of a word to its word vector representation
                    (e.g. GloVe or word2vec)
    :param topic_vectorizer: the CountVectorizer object used to transform
                             tokenized documents for LDA
    :param lda_model: the trained LDA model
    :param lowercase: lower-case the input examples for source and target
                      domains
    :return: the feature representations of the examples as a 2d numpy array of
             shape (num_examples, num_features)
    """
    features = np.zeros((len(examples), len(feature_names)))

    if lowercase:
        print('Lower-casing the source and target domain examples...')
        examples = [[word.lower() for word in example] for example in examples]
        trg_examples = [[word.lower() for word in trg_example]
                        for trg_example in trg_examples]

    # get the term distribution of the entire training data
    # (a numpy array of shape (vocab_size,) )
    train_term_dist = similarity.get_term_dist(examples, vocab)

    # get the term distribution of each training example
    # (a numpy array of shape (num_examples, vocab_size) )
    term_dists = np.array([similarity.get_term_dist(
        [example], vocab) for example in examples])

    # get the term distribution of the target data
    # (a numpy array of shape (vocab_size, ) )
    trg_term_dist = similarity.get_term_dist(trg_examples, vocab)

    # get the topic distributions of the train and target data
    topic_dists, trg_topic_dist = None, None
    if any(f_name.startswith('topic') for f_name in feature_names):
        print('Retrieving the topic distributions of source and target data...')
        topic_dists = similarity.get_topic_distributions(
            examples, topic_vectorizer, lda_model)
        trg_topic_dist = np.mean(similarity.get_topic_distributions(
            trg_examples, topic_vectorizer, lda_model), axis=0)
        print('Shape of topic distributions:', topic_dists.shape)
        print('Shape of target topic dist:', trg_topic_dist.shape)

    # get the word embedding representations of the train and target data
    word_reprs, trg_word_repr = None, None
    if any(f_name.startswith('word_embedding') for f_name in feature_names):
        print('Retrieving the word embedding representations of source and '
              'target data...')
        word_reprs = similarity.weighted_sum_of_embeddings(
            examples, vocab.word2id, word2vec, train_term_dist)
        trg_word_repr = np.mean(similarity.weighted_sum_of_embeddings(
            trg_examples, vocab.word2id, word2vec, trg_term_dist), axis=0)
        print('Shape of word representations:', word_reprs.shape)
        print('Shape of target word representation:', trg_word_repr.shape)

    for i in range(len(examples)):
        for j, f_name in enumerate(feature_names):
            # check whether feature belongs to similarity-based features,
            # diversity-based features, etc.
            if f_name.startswith('topic'):
                f = similarity.similarity_name2value(
                    f_name.split('_')[1], topic_dists[i], trg_topic_dist)
            elif f_name.startswith('word_embedding'):
                f = similarity.similarity_name2value(
                    f_name.split('_')[2], word_reprs[i], trg_word_repr)
            elif f_name in SIMILARITY_FUNCTIONS:
                f = similarity.similarity_name2value(
                    f_name, term_dists[i], trg_term_dist)
            elif f_name in DIVERSITY_FEATURES:
                f = diversity_feature_name2value(
                    f_name, examples[i], train_term_dist, vocab.word2id, word2vec)
            else:
                raise ValueError('%s is not a valid feature name.' % f_name)
            assert not np.isnan(f), 'Error: Feature %s is nan.' % f_name
            assert not np.isinf(f), 'Error: Feature %s is inf or -inf.' % f_name
            features[i, j] = f
        if i % 100 == 0 and i > 0:
            print('Created features for %d examples.' % i)
    return features


def get_feature_names(feature_set_names):
    """
    Given a list of feature sets, return the list of all their feature names.
    :param feature_set_names: a list of feature set names,
                              e.g. 'similarity', 'diversity', etc.
    :return: a list of feature names
    """
    features = []
    if 'similarity' in feature_set_names:
        print('Using similarity-based features...')
        features += SIMILARITY_FUNCTIONS
    if 'topic_similarity' in feature_set_names:
        print('Using similarity-based features with topic distributions...')
        features += ['topic_%s' % s for s in SIMILARITY_FUNCTIONS]
    if 'word_embedding_similarity' in feature_set_names:
        print('Using similarity-based features with word embedding '
              'representations...')
        # jensen-shannon, renyi, bhattacharyya do not work for word embeddings
        # as they require positive probabilities
        features += ['word_embedding_%s' % s for s in SIMILARITY_FUNCTIONS
                     if s not in ['jensen-shannon', 'renyi', 'bhattacharyya']]
    if 'diversity' in feature_set_names:
        print('Using diversity-based features...')
        features += DIVERSITY_FEATURES
    print('Using %d features.' % (len(features)))
    return features


# DIVERSITY-BASED FEATURES

def diversity_feature_name2value(f_name, example, train_term_dist, word2id,
                                 word2vec):
    """
    Given a feature name, return the corresponding feature value.
    :param f_name: the name of the feature
    :param example: the tokenised example document
    :param train_term_dist: the term distribution of the training data
    :param word2id: the word-to-id mapping
    :param word2vec: a mapping of a word to its word vector representation (e.g. GloVe or word2vec)
    :return: the value of the corresponding feature
    """
    if f_name == 'num_word_types':
        return number_of_word_types(example)
    if f_name == 'type_token_ratio':
        return type_token_ratio(example)
    if f_name == 'entropy':
        return entropy(example, train_term_dist, word2id)
    if f_name == 'simpsons_index':
        return simpsons_index(example, train_term_dist, word2id)
    if f_name == 'quadratic_entropy':
        return quadratic_entropy(example, train_term_dist, word2id, word2vec)
    if f_name == 'renyi_entropy':
        return renyi_entropy(example, train_term_dist, word2id)
    raise ValueError('%s is not a valid feature name.' % f_name)


def number_of_word_types(example):
    """Counts the number of word types of the example."""
    return len(set(example))


def type_token_ratio(example):
    """Calculates the type-token ratio of the example."""
    return number_of_word_types(example) / len(example)


def entropy(example, train_term_dist, word2id):
    """Calculates Entropy (https://en.wikipedia.org/wiki/Entropy_(information_theory))."""
    summed = 0
    for word in set(example):
        if word in word2id:
            p_word = train_term_dist[word2id[word]]
            summed += p_word * np.log(p_word)
    return - summed


def simpsons_index(example, train_term_dist, word2id):
    """Calculates Simpson's Index (https://en.wikipedia.org/wiki/Diversity_index#Simpson_index)."""
    score = np.sum([np.power(train_term_dist[word2id[word]], 2) if word in word2id else 0
                    for word in set(example)])
    return score


def quadratic_entropy(example, train_term_dist, word2id, word2vec):
    """Calculates Quadratic Entropy."""
    assert word2vec is not None, ('Error: Word vector representations have to '
                                  'be available for quadratic entropy.')
    summed = 0
    for word_1 in set(example):
        if word_1 not in word2id or word_1 not in word2vec:
            continue  # continue as the product will be 0
        for word_2 in set(example):
            if word_2 not in word2id or word_2 not in word2vec:
                continue  # continue as the product will be 0
            p_1 = train_term_dist[word2id[word_1]]
            p_2 = train_term_dist[word2id[word_2]]
            vec_1 = word2vec[word_1]
            vec_2 = word2vec[word_2]
            sim = similarity.cosine_similarity(vec_1, vec_2)
            summed += sim * p_1 * p_2
    return summed


def renyi_entropy(example, domain_term_dist, word2id):
    """Calculates Rényi Entropy (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)."""
    alpha = 0.99
    summed = np.sum([np.power(domain_term_dist[word2id[word]], alpha) if word in word2id else 0 for word in set(example)])
    if summed == 0:
        # 0 if none of the words appear in the dictionary;
        # set to a small constant == low prob instead
        summed = 0.0001
    score = 1 / (1 - alpha) * np.log(summed)
    return score
