#!/bin/env python3
# -*- coding: utf-8 -*-

# Simple structured perceptron tagger (bplank, parts by andersjo) - Language Proc 2
import argparse
import codecs
from collections import defaultdict, Counter
import json
import re
import numpy as np
import sys
import random

np.set_printoptions(precision=4)


def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            yield (current_words, current_tags)
            current_words = []
            current_tags = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        yield (current_words, current_tags)


def memoize(f):
    """
    helper function to be used as decorator to memoize features
    :param f:
    :return:
    """
    memo = {}
    def helper(*args):
        key = tuple(args[1:])
        try:
            return memo[key]
        except KeyError:
            memo[key] = f(*args)
            return memo[key]
    return helper


class StructuredPerceptron(object):
    """
    implements a structured perceptron as described in Collins 2002
    """

    def __init__(self, seed=1512141834):
        """
        initialize model
        :return:
        """
        self.feature_weights = defaultdict(float)
        self.tags = set()

        self.START = "__START__"
        self.END = "__END__"
        print("using seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)

    def fit(self, train_data, iterations=5, learning_rate=0.2):
        """
        read in a CoNLL file, extract emission features iterate over instances to train weight vector
        :param file_name:
        :return:
        """
        averaged_weights = Counter()

        for iteration in range(iterations):
            correct = 0
            total = 0.0
            sys.stderr.write('iteration %s\n************\n' % (iteration+1))

            for i, (words, tags) in enumerate(train_data):
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

                for tag in tags:
                    self.tags.add(tag)

                # get prediction
                prediction = self.decode(words)

                # derive global features
                global_gold_features = self.get_global_features(words, tags)
                global_prediction_features = self.get_global_features(words, prediction)

                # update weight vector
                for fid, count in global_gold_features.items():
                    self.feature_weights[fid] += learning_rate * count
                for fid, count in global_prediction_features.items():
                    self.feature_weights[fid] -= learning_rate * count

                # compute training accuracy for this iteration
                correct += sum([1 for (predicted, gold) in zip(prediction, tags) if predicted == gold])
                total += len(tags)

            #sys.stderr.write('\n\t%s features\n' % (len(self.feature_weights)))
            averaged_weights.update(self.feature_weights)
            sys.stderr.write('\tTraining accuracy: %.4f\n\n' % (correct/total))

            random.shuffle(train_data)

        self.feature_weights = averaged_weights

    def get_global_features(self, words, tags):
        """
        count how often each feature fired for the whole sentence
        :param words:
        :param tags:
        :return:
        """
        feature_counts = Counter()

        for i, (word, tag) in enumerate(zip(words, tags)):
            previous_tag = self.START if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag))

        return feature_counts

    @memoize
    def get_features(self, word, tag, previous_tag):
        """
        get all features that can be derived from the word and tags
        :param word:
        :param tag:
        :param previous_tag:
        :return:
        """
        word_lower = word.lower()
        prefix = word_lower[:3]
        suffix = word_lower[-3:]

        features = [
                    'TAG_%s' % (tag),                       # current tag
                    'TAG_BIGRAM_%s_%s' % (previous_tag, tag),  # tag bigrams
                    'WORD+TAG_%s_%s' % (word, tag),            # word-tag combination
                    'WORD_LOWER+TAG_%s_%s' % (word_lower, tag),# word-tag combination (lowercase)
                    'UPPER_%s_%s' % (word[0].isupper(), tag),  # word starts with uppercase letter
                    'DASH_%s_%s' % ('-' in word, tag),         # word contains a dash
                    'PREFIX+TAG_%s_%s' % (prefix, tag),        # prefix and tag
                    'SUFFIX+TAG_%s_%s' % (suffix, tag),        # suffix and tag

                    #########################
                    # ADD MOAAAAR FEATURES! #
                    #########################
                    ('WORDSHAPE', self.shape(word), tag),
                    'WORD+TAG_BIGRAM_%s_%s_%s' % (word, tag, previous_tag),
                    'SUFFIX+2TAGS_%s_%s_%s' % (suffix, previous_tag, tag),
                    'PREFIX+2TAGS_%s_%s_%s' % (prefix, previous_tag, tag)
        ]

        return features

    @memoize
    def shape(self, x):
        result = []
        for c in x:
            if c.isupper():
                result.append('X')
            elif c.islower():
                result.append('x')
            elif c in '0123456789':
                result.append('d')
        else:
            result.append(c)

        # replace multiple occurrences of a character with 'x*' and return it
        return re.sub(r"x+", "x*", ''.join(result))

    def decode(self,words):
        """
        Find best sequence
        :param words:
        :return:
        """
        N=len(words)
        M=len(self.tags) #number of tags
        tags=list(self.tags)

        # create trellis of size M (number of tags) x N (sentence length)
        Q = np.ones((len(self.tags), N)) * float('-Inf')
        backp = np.ones((len(self.tags), N), dtype=np.int16) * -1 #backpointers

        ### initialization step
        cur_word=words[0]
        for j in range(M):
            # initialize probs for tags j at position 1 (first word)
            cur_tag=tags[j]
            features = self.get_features(words[0], cur_tag, self.START)
            feature_weights = sum((self.feature_weights[x] for x in features))
            Q[j,0]=feature_weights

        # iteration step
        # filling the lattice, for every position and every tag find viterbi score Q
        for i in range(1,N):
            # for every tag
            for j in range(M):
                # checks if we are at end or start
                tag=tags[j]

                best_score = float('-Inf')

                # for every possible previous tag
                for k in range(M):

                    # k=previous tag
                    previous_tag=tags[k]

                    best_before=Q[k,i-1] # score until best step before

                    features = self.get_features(words[i], tag, previous_tag)
                    feature_weights = sum((self.feature_weights[x] for x in features))

                    score = best_before + feature_weights

                    if score > best_score:
                        Q[j,i]=score
                        best_score = score
                        backp[j,i]=k #best tag

        # final best
        #best_id=np.argmax(Q[:, -1]) #the same
        best_id=Q[:,-1].argmax()

        ## print best tags in reverse order
        predtags=[]
        predtags.append(tags[best_id])

        for i in range(N-1,0,-1):
            idx=int(backp[best_id,i])
            predtags.append(tags[idx])
            best_id=idx

        #return reversed predtags
        #return (words,predtags[::-1])
        return predtags[::-1]

    def predict(self, test_data):
        """
        Get predictions for entire test set
        :param test_data:
        :return:
        """
        return [self.decode(words) for words in test_data]

    def predict_eval(self, test_data, output=False):
        """
        compute accuracy on a test file
        :param file_name:
        :param output:
        :return:
        """
        correct = 0
        total = 0.0
        sys.stderr.write('\nTesting\n')
        sys.stderr.write('*******\n')

        for i, (words, tags) in enumerate(test_data):
            if i%100==0:
                sys.stderr.write('%s'%i)
            elif i%10==0:
                sys.stderr.write('.')

            # get prediction
            prediction = self.decode(words)

            if output:
                for word, gold, pred in zip(words, tags, prediction):
                    print("{}\t{}\t{}".format(word, gold, pred))
                print("")

            correct += sum([1 for (predicted, gold) in zip(prediction, tags) if predicted == gold])
            total += len(tags)
        print("\nTest accuracy on %s items: %.4f" % (i+1, correct/total), file=sys.stderr)

    def save(self, file_name):
        """
        save model
        :param file_name:
        :return:
        """
        print("saving model...", end=' ', file=sys.stderr)
        with codecs.open(file_name, "w", encoding='utf-8') as model:
            model.write("%s\n" % json.dumps({'tags': list(self.tags), 'weights': dict(self.feature_weights)}))
        print("done", file=sys.stderr)

    def load(self, file_name):
        """
        load model from JSON file
        :param file_name:
        :return:
        """
        print("loading model...", end=' ', file=sys.stderr)
        model_data = codecs.open(file_name, 'r', encoding='utf-8').readline().strip()
        model = json.loads(model_data)
        self.tags = set(model['tags'])
        self.feature_weights = model['weights']
        print("done", file=sys.stderr)


# if script is run from command line, automatically execute the following
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Run a structured perceptron""")
    parser.add_argument("--train", help="train model on a file (CoNLL format)", required=False)
    parser.add_argument("--test", help="test model on a file (CoNLL format)", required=False)
    parser.add_argument("--output", help="output predictions to stdout", required=False,action="store_true")
    parser.add_argument("--load", help="load model from JSON file", required=False)
    parser.add_argument("--save", help="save model as JSON file", required=False)
    parser.add_argument("--iterations", help="number of training iterations", required=False, default=5, type=int)
    parser.add_argument("--learning_rate", help="learning rate during training", required=False, default=0.2, type=float)
    args = parser.parse_args()

    # create new model
    sp = StructuredPerceptron()

    if args.load:
        sp.load(args.load)

    if args.train:
        train_data = list(read_conll_file(args.train))
        sp.fit(train_data, iterations=args.iterations, learning_rate=args.learning_rate)

    if args.save:
        sp.save(args.save)

    # check whether to show predictions
    if args.test:
        test_data = list(read_conll_file(args.test))
        sp.predict_eval(test_data, output=args.output)
