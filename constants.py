"""
Constants that are shared across files.
"""

NEG_ID = 0  # the negative sentiment id
POS_ID = 1  # the positive sentiment id
NEU_ID = 2  # the neutral sentiment id

# feature-related constants
FEATURE_SETS = ['similarity', 'topic_similarity', 'word_embedding_similarity',
                'diversity']
SIMILARITY_FUNCTIONS = ['jensen-shannon', 'renyi', 'cosine', 'euclidean',
                        'variational', 'bhattacharyya']
DIVERSITY_FEATURES = ['num_word_types', 'type_token_ratio', 'entropy',
                      'simpsons_index', 'quadratic_entropy', 'renyi_entropy']

# task-related constants
POS = 'pos'
POS_BILSTM = 'pos_bilstm'
SENTIMENT = 'sentiment'
PARSING = 'parsing'
TASKS = [POS, POS_BILSTM, SENTIMENT, PARSING]
POS_PARSING_TRG_DOMAINS = ['answers', 'emails', 'newsgroups', 'reviews', 'weblogs', 'wsj']
SENTIMENT_TRG_DOMAINS = ['books', 'dvd', 'electronics', 'kitchen']
TASK2TRAIN_EXAMPLES = {
    POS: 2000, POS_BILSTM: 2000, SENTIMENT: 1600, PARSING: 2000
}
TASK2DOMAINS = {
    POS: POS_PARSING_TRG_DOMAINS, POS_BILSTM: POS_PARSING_TRG_DOMAINS,
    SENTIMENT: SENTIMENT_TRG_DOMAINS, PARSING: POS_PARSING_TRG_DOMAINS
}

# method-related constants
BAYES_OPT = 'bayes-opt'
RANDOM = 'random'
MOST_SIMILAR_DOMAIN = 'most-similar-domain'
MOST_SIMILAR_EXAMPLES = 'most-similar-examples'
ALL_SOURCE_DATA = 'all-source-data'
BASELINES = [RANDOM, MOST_SIMILAR_DOMAIN, MOST_SIMILAR_EXAMPLES, ALL_SOURCE_DATA]
