"""
Methods for measuring domain similarity according to different metrics based on
different representations.
"""

import os

from sklearn.feature_extraction.text import CountVectorizer
import gensim

import numpy as np
import scipy.stats
import scipy.spatial.distance


# SIMILARITY MEASURES

def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def renyi_divergence(repr1, repr2, alpha=0.99):
    """Calculates Renyi divergence (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence)."""
    log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
    sim = 1 / (alpha - 1) * np.log(log_sum)
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def cosine_similarity(repr1, repr2):
    """Calculates cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    if repr1 is None or repr2 is None:
        return 0
    assert not (np.isnan(repr2).any() or np.isinf(repr2).any())
    assert not (np.isnan(repr1).any() or np.isinf(repr1).any())
    sim = 1 - scipy.spatial.distance.cosine(repr1, repr2)
    if np.isnan(sim):
        # the similarity is nan if no term in the document is in the vocabulary
        return 0
    return sim


def euclidean_distance(repr1, repr2):
    """Calculates Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance)."""
    sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
    return sim


def variational_distance(repr1, repr2):
    """Also known as L1 or Manhattan distance (https://en.wikipedia.org/wiki/Taxicab_geometry)."""
    sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
    return sim


def kl_divergence(repr1, repr2):
    """Calculates Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)."""
    sim = scipy.stats.entropy(repr1, repr2)
    return sim


def bhattacharyya_distance(repr1, repr2):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim


def similarity_name2value(s_name, repr1, repr2):
    """Given a similarity function name, return the corresponding similarity function value."""
    if s_name == 'jensen-shannon':
        return jensen_shannon_divergence(repr1, repr2)
    if s_name == 'renyi':
        return renyi_divergence(repr1, repr2)
    if s_name == 'cos' or s_name == 'cosine':
        return cosine_similarity(repr1, repr2)
    if s_name == 'euclidean':
        return euclidean_distance(repr1, repr2)
    if s_name == 'variational':
        return variational_distance(repr1, repr2)
    if s_name == 'kl':
        return kl_divergence(repr1, repr2)
    if s_name == 'bhattacharyya':
        return bhattacharyya_distance(repr1, repr2)
    raise ValueError('%s is not a valid feature name.' % s_name)


# TERM DISTRIBUTIONS

def get_domain_term_dists(term_dist_path, domain2data, vocab, lowercase=True):
    """
    Retrieves relative term distributions from the provided domains.
    :param term_dist_path: the path where the term distributions of the domains
                           should be saved
    :param domain2data: the mapping of domains to (labeled_examples, labels,
                        unlabeled_examples) tuples
    :param vocab: the Vocabulary object
    :param lowercase: lower-case the input data
    :return: a mapping of domains to their term distributions,
             i.e. a numpy array of shape (vocab_size,)
    """
    domain2term_dist = {}
    if os.path.exists(term_dist_path):
        print('Loading the term distributions from file...')
        with open(term_dist_path, 'r') as f:
            for line in f:
                domain, term_dist = line.strip().split('\t')
                term_dist = np.fromstring(term_dist, count=vocab.size, sep=' ')
                assert len(term_dist) == vocab.size,\
                    ('Length of term dist for %s should be %d, is %d.' %
                     (domain, vocab.size, len(term_dist)))
                assert np.round(np.sum(term_dist), 6) == 1,\
                    ('Sum of term distribution is %.6f instead of 1. The '
                     'vocabulary was likely created with a larger '
                     'max_vocab_size.' % np.sum(term_dist))
                domain2term_dist[domain] = term_dist
        assert set(domain2term_dist.keys()) == set(domain2data.keys()),\
            ('Term distributions are not saved for all domains: "%s" and "%s"'
             'are not equal.' % (' '.join(domain2term_dist.keys()),
                                 ' '.join(domain2data.keys())))
        return domain2term_dist

    if lowercase:
        print('Lower-casing the data for calculating the term distributions...')

    # get the term domain counts for the term distributions
    for domain, (examples, _, unlabeled_examples) in domain2data.items():
        domain2term_dist[domain] = get_term_dist(
            examples + unlabeled_examples, vocab, lowercase)

    print('Writing relative frequency distributions to %s...' % term_dist_path)
    with open(term_dist_path, 'w') as f:
        for domain, term_dist in domain2term_dist.items():
            f.write('%s\t%s\n' % (domain, ' '.join([str(c) for c in term_dist])))
    return domain2term_dist


def get_term_dist(docs, vocab, lowercase=True):
    """
    Calculates the term distribution of a list of documents.
    :param docs: a list of tokenized docs; can also contain a single document
    :param vocab: the Vocabulary object
    :param lowercase: lower-case the input data
    :return: the term distribution of the input documents,
             i.e. a numpy array of shape (vocab_size,)
    """
    term_dist = np.zeros(vocab.size)
    for doc in docs:
        for word in doc:
            if lowercase:
                word = word.lower()
            if word in vocab.word2id:
                term_dist[vocab.word2id[word]] += 1

    # normalize absolute freqs to obtain a relative frequency term distribution
    term_dist /= np.sum(term_dist)
    if np.isnan(np.sum(term_dist)):
        # the sum is nan if docs only contains one document and that document
        # has no words in the vocabulary
        term_dist = np.zeros(vocab.size)
    return term_dist


def get_most_similar_domain(trg_domain, domain2term_dists,
                            similarity_name='jensen-shannon'):
    """
    Given a target domain, retrieve the domain that is most similar to it
    according to some domain similarity measure (default: Jensen-Shannon
    divergence).
    :param trg_domain: the target domain
    :param domain2term_dists: a mapping of domain names to their term distribution
                              (a numpy array of shape (vocab_size,) )
    :param similarity_name: a string indicating the name of the similarity
                            measure used (default: 'jensen-shannon')
    :return: the domain most similar to the target domain
    """
    highest_sim_score, most_similar_domain = 0, None
    trg_term_dist = domain2term_dists[trg_domain]
    for domain, src_term_dist in domain2term_dists.items():
        if domain == trg_domain:
            continue
        sim_score = similarity_name2value(similarity_name, src_term_dist, trg_term_dist)
        if sim_score > highest_sim_score:
            highest_sim_score, most_similar_domain = sim_score, domain
    return most_similar_domain


# TOPIC DISTRIBUTIONS

def train_topic_model(examples, vocab, num_topics=50, num_iterations=2000,
                      num_passes=10):
    """
    Trains an LDA topic model on the provided list of tokenised documents and
    returns the vectorizer used for the transformation and the trained LDA
    model.
    :param examples: a list of tokenised documents of all domains
    :param vocab: the Vocabulary object
    :param num_topics: the number of topics that should be used
    :param num_iterations: the number of iterations
    :param num_passes: the number of passes over the corpus that should be
                       performed
    :return: the CountVectorizer used for transforming the corpus and the
             trained LDA topic model
    """
    # the text is already tokenized and pre-processed; we only need to
    # transform it to vectors
    vectorizer = CountVectorizer(vocabulary=vocab.word2id,
                                 tokenizer=lambda x: x,
                                 preprocessor=lambda x: x)
    lda_corpus = vectorizer.fit_transform(examples)

    # the gensim LDA implementation requires a sparse corpus;
    # we could also use sci-kit learn instead
    lda_corpus = gensim.matutils.Sparse2Corpus(lda_corpus,
                                               documents_columns=False)
    print('Training LDA model on data of all domains with %d topics, '
          '%d iterations, %d passes...' % (num_topics, num_iterations,
                                           num_passes))
    lda_model = gensim.models.LdaMulticore(
        lda_corpus, num_topics=num_topics, id2word=vocab.id2word,
        iterations=num_iterations, passes=num_passes)
    return vectorizer, lda_model


def get_topic_distributions(examples, vectorizer, lda_model):
    """
    Retrieve the topic distributions of a collection of documents.
    :param examples: a list of tokenised documents
    :param vectorizer: the CountVectorizer used for transforming the documents
    :param lda_model: the trained LDA model
    :return: an array of shape (num_examples, num_topics) containing the topic
             distribution of each example
    """
    vectorized_corpus = vectorizer.transform(examples)
    gensim_corpus = gensim.matutils.Sparse2Corpus(vectorized_corpus,
                                                  documents_columns=False)
    topic_representations = []
    for doc in gensim_corpus:
        topic_representations.append(
            [topic_prob for (topic_id, topic_prob) in
             lda_model.get_document_topics(doc, minimum_probability=0.)])
    return np.array(topic_representations)


# PRE-TRAINED WORD EMBEDDINGS METHODS

def load_word_vectors(file, vocab_word_vec_file, word2id, vector_size=300,
                      header=False):
    """
    Loads word vectors from a text file, e.g. the one obtained from
    http://nlp.stanford.edu/projects/glove/.
    :param file: the file the word vectors should be loaded from
    :param vocab_word_vec_file: the file where the word embeddings in the
                                vocabulary can be stored for faster retrieval
    :param word2id: the mapping of words to their ids in the vocabulary
    :param vector_size: the size of the word vectors
    :param header: whether the word vectors text file contains a header;
                   default is False
    :return a dictionary mapping each word to its numpy word vector
    """
    word2vector = {}
    if os.path.exists(vocab_word_vec_file):
        print('Loading vocabulary word vectors from %s...' % vocab_word_vec_file)
        with open(vocab_word_vec_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.split(' ')[0]
                assert word in word2id, ('Error: %s in vocab word vec file is '
                                         'not in vocab.' % word)
                line = ' '.join(line.split(' ')[1:]).strip()
                vector = np.fromstring(line, dtype=float, sep=' ')
                assert len(vector) == vector_size,\
                    ('Error: %d != vector size %d for word %s.'
                     % (len(vector), vector_size, word))
                word2vector[word] = vector
        return word2vector

    print('Reading word vectors from %s...' % file)
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and header:
                continue
            if i % 100000 == 0 and i > 0:
                print('Processed %d vectors.' % i)
            word = line.split(' ')[0]
            if word not in word2id:
                continue
            line = ' '.join(line.split(' ')[1:]).strip()
            vector = np.fromstring(line, dtype=float, sep=' ')
            assert len(vector) == vector_size
            word2vector[word] = vector

    print('Writing word vectors to %s...' % vocab_word_vec_file)
    with open(vocab_word_vec_file, 'w', encoding='utf-8') as f:
        for word, vector in word2vector.items():
            f.write('%s %s\n' % (word, ' '.join([str(c) for c in vector])))
    return word2vector


def weighted_sum_of_embeddings(docs, word2id, word2vector, term_dist):
    """
    Get a weighted sum of embeddings representation for a list of documents
    belonging to one domain. The documents are represented as a list of
    ngrams. Also works if the list only contains a single document.
    :param docs: a list of documents
    :param word2id: the mapping of words to their ids in the vocabulary
    :param word2vector: the mapping of words to their vector representations
    :param term_dist: the term distribution of the data the words belong to
    :return: the vector representation of the provided list of documents
    """
    # the factor with which the word probability is smoothed, we empirically
    # set this to the value used in Mikolov et al. (2013)
    t = 10e-5
    word_embed_representations = []
    for doc in docs:
        doc_vector = np.zeros(len(list(word2vector.values())[0]))
        word_vector_count = 0
        for word in doc:
            if word in word2vector:
                vector = word2vector[word]

                # weight the vector with the smoothed inverse probability of
                # the word
                doc_vector += np.sqrt(t / (term_dist[word2id[word]])) * vector
                word_vector_count += 1
        if word_vector_count == 0:
            # this might be because the review is in another language by
            # accident; set count to 1 to avoid division by 0
            word_vector_count = 1
        doc_vector /= word_vector_count
        assert not (np.isnan(doc_vector).any() or np.isinf(doc_vector).any())
        word_embed_representations.append(doc_vector)
    return np.array(word_embed_representations)
