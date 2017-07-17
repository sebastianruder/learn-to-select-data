#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility methods for loading and processing data.
"""

import os
import codecs
from collections import Counter
import itertools
import operator

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import NEG_ID, POS_ID
from simpletagger import read_conll_file

from constants import SENTIMENT, POS, POS_BILSTM, PARSING, \
    SENTIMENT_TRG_DOMAINS, POS_PARSING_TRG_DOMAINS
from bist_parser.bmstparser.src.utils import read_conll


class Vocab:
    """
    The vocabulary class. Stores the word-to-id mapping.
    """
    def __init__(self, max_vocab_size, vocab_path):
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.size = 0
        self.word2id = {}
        self.id2word = {}

    def load(self):
        """
        Loads the vocabulary from the vocabulary path.
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Reading vocabulary from %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_vocab_size:
                    print('Vocab in file is larger than max vocab size. '
                          'Only using top %d words.' % self.max_vocab_size)
                    break
                word, idx = line.split('\t')
                self.word2id[word] = int(idx.strip())
        self.size = len(self.word2id)
        self.id2word = {index: word for word, index in self.word2id.items()}
        assert self.size <= self.max_vocab_size, \
            'Loaded vocab is of size %d., max vocab size is %d.' % (
                self.size, self.max_vocab_size)

    def create(self, texts, lowercase=True):
        """
        Creates the vocabulary and stores it at the vocabulary path.
        :param texts: a list of lists of tokens
        :param lowercase: lowercase the input texts
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Building the vocabulary...')
        if lowercase:
            print('Lower-casing the input texts...')
            texts = [[word.lower() for word in text] for text in texts]

        word_counts = Counter(itertools.chain(*texts))

        # get the n most common words
        most_common = word_counts.most_common(n=self.max_vocab_size)

        # construct the word to index mapping
        self.word2id = {word: index for index, (word, count)
                        in enumerate(most_common)}
        self.id2word = {index: word for word, index in self.word2id.items()}

        print('Writing vocabulary to %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word, index in sorted(self.word2id.items(),
                                      key=operator.itemgetter(1)):
                f.write('%s\t%d\n' % (word, index))
        self.size = len(self.word2id)


def get_all_docs(domain_data_pairs, unlabeled=True):
    """
    Return all labeled and undocumented documents of multiple domains.
    :param domain_data_pairs: a list of (domain, (labeled_reviews, labels,
                              unlabeled_reviews)) tuples as obtained by
                              domain2data.items()
    :param unlabeled: whether unlabeled documents should be incorporated
    :return: a list containing the documents from all domains, the corresponding
             labels, and a list containing the domain of each example
    """
    docs, labels, domains = [], [], []
    for domain, (labeled_docs, doc_labels, unlabeled_docs) in domain_data_pairs:
        length_of_docs = 0
        if not scipy.sparse.issparse(labeled_docs):
            # if the labeled documents are not a sparse matrix, i.e.
            # a tf-idf matrix, we can just flatten them into one array
            docs += labeled_docs
            length_of_docs += len(labeled_docs)
            if unlabeled:
                # if specified, we add the unlabeled documents
                docs += unlabeled_docs
                length_of_docs += len(labeled_docs)
        else:
            # if it is a sparse matrix, we just append the docs as a list and
            # then stack the list in the end
            docs.append(labeled_docs)
            length_of_docs += labeled_docs.shape[0]
            if unlabeled and unlabeled_docs is not None:
                docs.append(unlabeled_docs)
                length_of_docs += unlabeled_docs.shape[0]
        labels.append(doc_labels)

        # we just add the corresponding domain for each document so that we can
        # later see where the docs came from
        domains += [domain] * length_of_docs
    if scipy.sparse.issparse(labeled_docs):
        # finally, if the matrix was sparse, we can stack the documents together
        docs = scipy.sparse.vstack(docs)
    return docs, np.hstack(labels), domains


def get_tfidf_data(domain2data, vocab):
    """
    Transform the tokenized documents of each domain into a tf-idf matrix.
    :param domain2data: the mapping of domains to a (tokenized_reviews, labels,
                        tokenized_unlabeled_reviews) tuple
    :param vocab: the Vocabulary class
    :return: a mapping of domains to a (labeled_tfidf_matrix, labels,
             unlabeled_tfidf_matrix) tuple where both tfidf matrices are
             scipy.sparse.csr.csr_matrix with shape (num_examples, vocab_size)
    """
    domain2tfidf_data = {}
    for domain, (labeled_examples, labels, unlabeled_examples) in domain2data.items():

        # apply the vectorizer to the already tokenized and pre-processed input
        vectorizer = TfidfVectorizer(vocabulary=vocab.word2id,
                                     tokenizer=lambda x: x,
                                     preprocessor=lambda x: x)

        # fit the vectorizer to both labeled and unlabeled examples but keep
        # the transformed examples separate
        vectorizer.fit(labeled_examples + unlabeled_examples)
        tfidf_labeled_examples = vectorizer.transform(labeled_examples)

        # note: we cap unlabeled examples at 100k (only relevant for the books
        # domain in the large-scale setting)
        unlabeled_examples = unlabeled_examples[:100000]
        tfidf_unlabeled_examples = vectorizer.transform(unlabeled_examples) \
            if len(unlabeled_examples) != 0 else None
        assert isinstance(tfidf_labeled_examples, scipy.sparse.csr.csr_matrix),\
            'The input is not a sparse matrix.'
        assert isinstance(labels, np.ndarray), 'Labels are not a numpy array.'
        domain2tfidf_data[domain] = [tfidf_labeled_examples, labels,
                                     tfidf_unlabeled_examples]
    return domain2tfidf_data

  
def log_to_file(log_file, run_dict, trg_domain, args):
    """
    Log the results of experiment runs to a file.
    :param log_file: the file used for logging
    :param run_dict: a dictionary mapping a method name to a list of
                     (val_accuracy, test_accuracy) tuples or a list
                     of (val_accuracy, test_accuracy, best_feature_weight)
                     tuples for the bayes-opt method
    :param trg_domain: the target domain
    :param args: the arguments used as input to the script
    """
    with open(log_file, 'a') as f:
        for method, scores in run_dict.items():
            best_feature_weights = ''
            if len(scores) == 0:
                continue
            if method.startswith('bayes-opt'):
                val_accuracies, test_accuracies, best_feature_weights = \
                    zip(*scores)
            else:
                val_accuracies, test_accuracies = zip(*scores)
            mean_val, std_val = np.mean(val_accuracies), np.std(val_accuracies)
            mean_test, std_test = np.mean(test_accuracies),\
                                  np.std(test_accuracies)
            # target domain. method. feature_sets.  # all other params
            f.write('%s\t%s\t%s\t%.4f (+-%.4f)\t%.4f (+-%.4f)\t[%s]\t[%s]\t%s\t'
                    '%s\n'
                    % (trg_domain, method, ' '.join(args.feature_sets),
                       mean_val, std_val, mean_test, std_test,
                       ', '.join(['%.4f' % v for v in val_accuracies]),
                       ', '.join(['%.4f' % t for t in test_accuracies]),
                       str(list(best_feature_weights)),
                       ' '.join(['%s=%s' % (arg, str(getattr(args, arg)))
                                 for arg in vars(args)])))


def read_feature_weights_file(feature_weights_path):
    """
    Reads a manually created file containing the learned feature weights for
    some task, trg domain, and feature set and returns them.
    The file format is this (note that ~ is used as delimiter to avoid clash
    with other delimiters in the feature sets):
    books~similarity diversity~[0.0, -0.66, -0.66, 0.66, 0.66, -0.66, 0.66, 0.0, 0.0, -0.66, 0.66, 0.66]
    ...
    :param feature_weights_path: the path to the feature weights file
    :return: a generator of tuples (feature_weights_domain, feature_set, feature_weights)
    """
    print('Reading feature weights from %s...' % feature_weights_path)
    with open(feature_weights_path, 'r') as f:
        for line in f:
            feature_weights_domain, feature_set, feature_weights =\
                line.split('~')
            feature_weights = feature_weights.strip('[]\n')
            feature_weights = feature_weights.split(', ')
            feature_weights = [float(f) for f in feature_weights]
            print('Feature weights domain: %s. Feature set: %s. '
                  'Feature weights: %s' %
                  (feature_weights_domain, feature_set, str(feature_weights)))
            yield feature_weights_domain, feature_set, feature_weights


def task2read_data_func(task):
    """Returns the read data method for each task."""
    if task == SENTIMENT:
        return read_processed
    if task in [POS, POS_BILSTM]:
        return read_tagging_data
    if task == PARSING:
        return read_parsing_data
    raise ValueError(
        'No data reading function available for task %s.' % task)


# =============== sentiment data functions =======

def read_processed(dir_path):
    """
    Reads the processed files in the processed_acl directory.
    :param dir_path: the directory containing the processed_acl folder
    :return: a dictionary that maps domains to a tuple of
             (labeled_reviews,labels, unlabeled_reviews); labeled_reviews is
             a list of reviews where each review is a list of (unordered)
             ngrams; labels is a numpy array of label ids of shape (num_labels);
             unlabeled_reviews has the same format as labeled_reviews
    """
    domains_path = os.path.join(dir_path, 'processed_acl')
    assert os.path.exists(domains_path), ('Error: %s does not exist.' %
                                          domains_path)
    domains = os.listdir(domains_path)
    assert set(domains) == set(SENTIMENT_TRG_DOMAINS)
    domain2data = {domain: [[], [], None] for domain in domains}
    for domain in domains:
        print('Processing %s...' % domain)
        # file names are positive.review, negative.review, and unlabeled.review
        # positive and negative each contain 2k examples;
        # unlabeled contains ~4k examples
        splits = ['positive', 'negative', 'unlabeled']
        for split in splits:
            print('Processing %s/%s...' % (domain, split), end='')
            file_path = os.path.join(domains_path, domain, '%s.review' % split)
            assert os.path.exists(file_path), '%s does not exist.' % file_path
            reviews = []
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    # get the pre-processed features; these are a white-space
                    # separated list of unigram/bigram occurrence counts in
                    # the document, e.g. "must:1", "still_has:1"
                    features = line.split(' ')[:-1]

                    # convert the features to a sequence (note: order does not
                    # matter here); we do this to be able to later use the
                    # same post-processing as for data from other sources
                    review = []
                    for feature in features:
                        ngram, count = feature.split(':')
                        for _ in range(int(count)):
                            review.append(ngram)

                    # add the review to the reviews
                    reviews.append(review)

            # the domain2data dict maps a domain to a tuple of
            # (reviews, labels, unlabeled_reviews)
            if split == 'unlabeled':
                # add the unlabeled reviews at the third position of the tuple
                domain2data[domain][2] = reviews
            else:
                # add labels with the same polarity as the file
                domain2data[domain][0] += reviews
                domain2data[domain][1] += [sentiment2id(split)] * len(reviews)

            print(' Processed %d reviews.' % len(reviews))
        domain2data[domain][1] = np.array(domain2data[domain][1])
    return domain2data


def sentiment2id(sentiment):
    """
    Maps a sentiment to a label id.
    :param sentiment: the sentiment; one of [positive, pos, negative, neg]
    :return: the id of the specified sentiment
    """
    if sentiment in ['positive', 'pos']:
        return POS_ID
    if sentiment in ['negative', 'neg']:
        return NEG_ID
    raise ValueError('%s is not a valid sentiment.' % sentiment)


# =============== tagging data functions ======

def read_tagging_data(dir_path, top_k_unlabeled=2000):
    """
    Reads the CoNLL tagging files in the gweb_sancl/pos directory. Outputs the
    documents as list of lists with tokens and lists of corresponding tags.
    The domains are reviews, answer, emails, newsblogs, weblogs, wsj and
    the corresponding files are called gweb-{domain}-{dev|test}.conll in folder
    gweb_sancl/pos/{domain}
    :param dir_path: the path to the directory gweb_sancl
    :param top_k_unlabeled: only use the top k unlabeled examples
    :return: a dictionary that maps domains to a tuple of (labeled_examples,
             labels, unlabeled_examples); labeled_examples is a list of
             sentences where each sentence is a list of tokens; labels
             is a list of tags for each sentence; unlabeled_examples has the
             same format as labeled_examples
    """
    domains_path = os.path.join(dir_path, 'pos')
    assert os.path.exists(domains_path), ('Error: %s does not exist.' %
                                         domains_path)
    domains = [d for d in os.listdir(domains_path)]
    print(domains)
    assert set(domains) == set(POS_PARSING_TRG_DOMAINS)
    domain2data = {domain: [[], [], None] for domain in domains}
    for domain in domains:
        print('Processing %s...' % domain)
        # file names are pos/{domain}/gweb-{domain}-{dev|test}.conll
        splits = ['dev', 'test', 'unlabeled']
        for split in splits:
            print('Processing %s/%s...' % (domain, split), end='')

            if split == 'unlabeled':
                file_path = os.path.join(dir_path, 'unlabeled',
                                         'gweb-%s.unlabeled.txt' % (domain))
                assert os.path.exists(file_path), ('%s does not exist.' %
                                                   file_path)
                unlabeled_data = []
                print(file_path)
                with open(file_path,'rb') as f:
                    for line in f:
                        line = line.decode('utf-8','ignore').strip().split()
                        unlabeled_data.append(line)
                # add the unlabeled reviews at the third position of the tuple
                print('Read %s number of unlabeled sentences'
                      % len(unlabeled_data))

                unlabeled_data = unlabeled_data[:top_k_unlabeled]
                print('Took top {} documents '.format(top_k_unlabeled))
                domain2data[domain][2] = unlabeled_data
            else:

                file_path = os.path.join(domains_path, domain,
                                         'gweb-%s-%s.conll' % (domain, split))
                assert os.path.exists(file_path), ('%s does not exist.' %
                                                   file_path)

                data = list(read_conll_file(file_path))
                words = [words for words, tags in data]
                tags = [tags for words, tags in data]
                domain2data[domain][0] += words
                domain2data[domain][1] += tags

            print(' Processed %d sentences.' % len(data))
        domain2data[domain][1] = np.array(domain2data[domain][1])
    return domain2data


# =============== parsing data functions ======

def read_parsing_data(dir_path, top_k_unlabeled=2000):
    """
    Reads the CoNLL parsing files in the gweb_sancl/pos directory
    :param dir_path: The gweb_sancl directory path.
    :param top_k_unlabeled: only use the top k unlabeled examples
    :return: a dictionary that maps domains to a tuple of (
             labeled_conll_entries, pseudo_labels, unlabeled_conll_entries);
             labeled_conll_entries is a list of CoNLLEntry containing the
             word forms, annotations, and target labels to be used for
             parsing; since each CoNLLEntry already contains the target label,
             pseudo_labels only contains pseudo-labels; unlabeled_conll_entries
             are used as unlabeled data
    """
    domains_path = os.path.join(dir_path, 'parse')
    assert os.path.exists(domains_path), ('Error: %s does not exist.' %
                                          domains_path)
    domains = [d for d in os.listdir(domains_path)]
    print(domains)
    assert set(domains) == set(POS_PARSING_TRG_DOMAINS)
    domain2data = {domain: [[], [], None] for domain in domains}
    for domain in domains:
        print('Processing %s...' % domain)
        # file names are pos/{domain}/gweb-{domain}-{dev|test}.conll
        splits = ['dev', 'test', 'unlabeled']
        for split in splits:
            print('Processing %s/%s...' % (domain, split), end='')
            if split == 'unlabeled':
                file_path = os.path.join(dir_path, 'unlabeled',
                                         'gweb-%s.unlabeled.txt' % (domain))
                assert os.path.exists(file_path), ('%s does not exist.' %
                                                   file_path)
                unlabeled_data = []
                with open(file_path,'rb') as f:
                    for line in f:
                        line = line.decode('utf-8','ignore').strip().split()
                        unlabeled_data.append(line)

                # add the unlabeled reviews at the third position of the tuple
                print('Read %s number of unlabeled sentences' % len(unlabeled_data))

                unlabeled_data = unlabeled_data[:top_k_unlabeled]
                print('Took top {} documents '.format(top_k_unlabeled))
                domain2data[domain][2] = unlabeled_data
            else:
                if domain == 'wsj' and split == 'test':
                    file_path = os.path.join(domains_path, domain,
                                             'ontonotes-%s-%s.conll'
                                             % (domain, split))
                else:
                    file_path = os.path.join(domains_path, domain,
                                             'gweb-%s-%s.conll'
                                             % (domain, split))
                assert os.path.exists(file_path), ('%s does not exist.' %
                                                   file_path)

                with open(file_path, 'r') as conll_file_path:
                    data = list(read_conll(conll_file_path))
                domain2data[domain][0] += data

                # add pseudo-labels since the model doesn't use explicit
                # labels for training
                domain2data[domain][1] += [0] * len(data)
        domain2data[domain][1] = np.array(domain2data[domain][1])
    return domain2data


def read_parsing_evaluation(evaluation_file_path):
    """
    Read the labeled attachment score, unlabeled attachment score, and label
    accuracy score from a file produced by the parsing evaluation perl
    script. The beginning of the file looks like this:
    Labeled   attachment score: 6995 / 9615 * 100 = 72.75 %
    Unlabeled attachment score: 7472 / 9615 * 100 = 77.71 %
    Label accuracy score:       8038 / 9615 * 100 = 83.60 %
    ...
    :param evaluation_file_path: the path of the evaluation file produced by the perl script
    :return: the labeled attachment score, the unlabeled attachment score, and the label accuracy score
    """
    try:
        with open(evaluation_file_path, 'r') as f:
            lines = f.readlines()
            las = float(lines[0].split('=')[1].strip('% \n'))
            uas = float(lines[1].split('=')[1].strip('% \n'))
            acc = float(lines[2].split('=')[1].strip('% \n'))
    except Exception:
        las = 0.0
        uas = 0.0
        acc = 0.0
    return las, uas, acc
