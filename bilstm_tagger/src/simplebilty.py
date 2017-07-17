#!/usr/bin/env python3
# coding=utf-8
"""
A neural network based tagger  (bi-LSTM) - version w/o MTL
:author: Barbara Plank
"""
import argparse
import random
import time
import sys
import numpy as np
import os
import pickle
import dynet

from bilstm_tagger.src.lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor
from bilstm_tagger.src.lib.mio import read_conll_file, load_embeddings_file


def main():
    parser = argparse.ArgumentParser(description="""Run the NN tagger""")
    parser.add_argument("--train",
                        help="train data")  # allow multiple train files, each asociated with a task = position in the list
    # parser.add_argument("--pred_layer", help="layer of predictons", default=1) # assume always h_layer here
    parser.add_argument("--model", help="load model from file", required=False)
    parser.add_argument("--iters", help="training iterations [default: 30]", required=False, type=int, default=30)
    parser.add_argument("--in_dim", help="input dimension [default: 64] (like Polyglot embeds)", required=False,
                        type=int, default=64)
    parser.add_argument("--c_in_dim", help="input dimension for character embeddings [default: 100]", required=False,
                        type=int, default=100)
    parser.add_argument("--h_dim", help="hidden dimension [default: 100]", required=False, type=int, default=100)
    parser.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False,
                        type=int, default=1)
    parser.add_argument("--test", nargs='*', help="test file(s)",
                        required=False)  # should be in the same order/task as train
    parser.add_argument("--dev", help="dev file(s)", required=False)
    parser.add_argument("--output", help="output predictions to file", required=False, default=None)
    parser.add_argument("--lower", help="lowercase words (not used)", required=False, default=False,
                        action="store_true")
    parser.add_argument("--save", help="save model to file (appends .model as well as .pickle)", required=False,
                        default=None)
    parser.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    parser.add_argument("--sigma", help="noise sigma", required=False, default=0.2, type=float)
    parser.add_argument("--ac", help="activation function [rectify, tanh, ...]", default="tanh",
                        type=MyNNTaggerArgumentOptions.acfunct)
    parser.add_argument("--trainer", help="trainer [sgd, adam] default: sgd", required=False, default="sgd")
    parser.add_argument("--dynet-seed", help="random seed for dynet (needs to be first argument!)", required=False,
                        type=int)
    parser.add_argument("--dynet-mem", help="memory for dynet (needs to be first argument!)", required=False, type=int)
    parser.add_argument("--save-embeds", help="save word embeddings file", required=False, default=None)

    args = parser.parse_args()

    if args.save:
        # check if folder exists
        if os.path.isdir(args.save):
            modeldir = os.path.dirname(args.save)
            if not os.path.exists(modeldir):
                os.makedirs(modeldir)
    if args.output:
        if os.path.isdir(args.output):
            outdir = os.path.dirname(args.output)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    start = time.time()

    if args.model:
        print("loading model from file {}".format(args.model), file=sys.stderr)
        tagger = load(args.model)
    else:
        tagger = SimpleBiltyTagger(args.in_dim,
                                   args.h_dim,
                                   args.c_in_dim,
                                   args.h_layers,
                                   embeds_file=args.embeds,
                                   activation=args.ac,
                                   lower=args.lower,
                                   noise_sigma=args.sigma)

    if args.train:
        ## read data
        train_X, train_Y = tagger.get_train_data(args.train)

        if dev:
            dev_X, dev_Y, org_X, org_Y, task_labels = self.get_data_as_indices(dev, "task0")

        tagger.fit(args.train, args.iters, args.trainer, seed=args.dynet_seed)
        if args.save:
            save(tagger, args.save)

    if args.test and len(args.test) != 0:
        stdout = sys.stdout
        # One file per test ... 
        for i, test in enumerate(args.test):
            if args.output != None:
                file_pred = args.output + ".task" + str(i)
                sys.stdout = open(file_pred, 'w')

            sys.stderr.write('\nTesting Task' + str(i) + '\n')
            sys.stderr.write('*******\n')
            test_X, test_Y, org_X, org_Y, task_labels = tagger.get_data_as_indices(test, "task" + str(i))
            correct, total = tagger.evaluate(test_X, test_Y, org_X, org_Y, task_labels, output_predictions=args.output)

            print("\ntask%s test accuracy on %s items: %.4f" % (i, i + 1, correct / total), file=sys.stderr)
            print(("Task" + str(i) + " Done. Took {0:.2f} seconds.".format(time.time() - start)), file=sys.stderr)
            sys.stdout = stdout

    if args.ac:
        activation = args.ac.__name__
    else:
        activation = "None"
    print("Info: biLSTM\n\tin_dim: {0}\n\tc_in_dim: {7}\n\th_dim: {1}"
          "\n\th_layers: {2}\n\tactivation: {4}\n\tsigma: {5}\n\tlower: {6}"
          "\tembeds: {3}".format(args.in_dim, args.h_dim, args.h_layers, args.embeds, activation, args.sigma,
                                 args.lower, args.c_in_dim), file=sys.stderr)

    if args.save_embeds:
        tagger.save_embeds(args.save_embeds)


def load(model_file):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(model_file + ".pickle", "rb"))
    tagger = SimpleBiltyTagger(myparams["in_dim"],
                               myparams["h_dim"],
                               myparams["c_in_dim"],
                               myparams["h_layers"],
                               activation=myparams["activation"])
    tagger.set_indices(myparams["w2i"], myparams["c2i"], myparams["tag2idx"])
    tagger.predictors, tagger.char_rnn, tagger.wembeds, tagger.cembeds = \
        tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])
    tagger.model.load(model_file)
    print("model loaded: {}".format(model_file), file=sys.stderr)
    return tagger


def save(nntagger, model_file_name):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    nntagger.model.save(model_file_name)
    import pickle
    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "tag2idx": nntagger.tag2idx,
                "activation": nntagger.activation,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "h_layers": nntagger.h_layers,
                "embeds_file": nntagger.embeds_file,
                "pred_layer": nntagger.pred_layer
                }
    pickle.dump(myparams, open(model_file_name + ".pickle", "wb"))
    print("model stored: {}".format(model_file_name), file=sys.stderr)


class SimpleBiltyTagger(object):
    def __init__(self, in_dim, h_dim, c_in_dim, h_layers, embeds_file=None, activation=dynet.tanh, lower=False,
                 noise_sigma=0.1, tasks_ids=[]):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.tag2idx = {}  # tag to tag_id mapping
        self.pred_layer = 1  # at which layer to predict
        self.model = dynet.Model()  # init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.lower = lower
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {},
                           "task_expected_at": {}}  # the inner layers and predictors
        self.wembeds = None  # lookup: embeddings for words
        self.cembeds = None  # lookup: embeddings for characters
        self.embeds_file = embeds_file
        self.char_rnn = None  # RNN for character input

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def set_indices(self, w2i, c2i, tag2idx):
        self.tag2idx = tag2idx
        self.w2i = w2i
        self.c2i = c2i

    def fit(self, train_X, train_Y, num_epochs, train_algo, val_X=None, val_Y=None, patience=2, model_path=None,
            seed=None):
        """
        train the tagger
        """
        print("read training data", file=sys.stderr)

        if seed:
            print(">>> using seed: ", seed, file=sys.stderr)
            random.seed(seed)  # setting random seed

        # init lookup parameters and define graph
        print("build graph", file=sys.stderr)

        num_words = len(self.w2i)
        num_chars = len(self.c2i)

        self.predictors, self.char_rnn, self.wembeds, self.cembeds = self.build_computation_graph(num_words, num_chars)

        if train_algo == "sgd":
            trainer = dynet.SimpleSGDTrainer(self.model)
        elif train_algo == "adam":
            trainer = dynet.AdamTrainer(self.model)
        else:
            raise ValueError('%s is not a valid optimizer.' % train_algo)

        assert (len(train_X) == len(train_Y))
        train_data = list(zip(train_X, train_Y))

        print('Starting training for %d epochs...' % num_epochs)
        best_val_acc, epochs_no_improvement = 0., 0
        if val_X is not None and val_Y is not None and model_path is not None:
            print('Using early stopping with patience of %d...' % patience)
        for cur_iter in range(num_epochs):
            total_loss = 0.0
            total_tagged = 0.0
            random.shuffle(train_data)
            for ((word_indices, char_indices), y) in train_data:
                # use same predict function for training and testing
                output = self.predict(word_indices, char_indices, train=True)

                loss1 = dynet.esum([self.pick_neg_log(pred, gold) for pred, gold in zip(output, y)])
                lv = loss1.value()
                total_loss += lv
                total_tagged += len(word_indices)

                loss1.backward()
                trainer.update()
            total_loss = total_loss / total_tagged
            print("epoch {2} {0:>12}: {1:.2f}".format("total loss", total_loss, cur_iter))

            # get the best accuracy on the validation set
            val_correct, val_total = self.evaluate(val_X, val_Y)
            val_accuracy = val_correct / val_total

            if val_X is not None and val_Y is not None and model_path is not None:
                if val_accuracy > best_val_acc:
                    print('Accuracy %.4f is better than best val accuracy %.4f.' % (val_accuracy, best_val_acc))
                    best_val_acc = val_accuracy
                    epochs_no_improvement = 0
                    save(self, model_path)
                else:
                    print('Accuracy %.4f is worse than best val loss %.4f.' % (val_accuracy, best_val_acc))
                    epochs_no_improvement += 1
                if epochs_no_improvement == patience:
                    print('No improvement for %d epochs. Early stopping...' % epochs_no_improvement)
                    break

    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        """
        # initialize the word embeddings and the parameters
        if self.embeds_file:
            print("loading embeddings", file=sys.stderr)
            embeddings, emb_dim = load_embeddings_file(self.embeds_file, lower=self.lower)
            assert (emb_dim == self.in_dim)
            num_words = len(set(embeddings.keys()).union(set(self.w2i.keys())))  # initialize all with embeddings
            # init model parameters and initialize them
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim))
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim))

            init = 0
            l = len(embeddings.keys())
            for word in embeddings.keys():
                # for those words we have already in w2i, update vector, otherwise add to w2i (since we keep data as integers)
                if word in self.w2i:
                    wembeds.init_row(self.w2i[word], embeddings[word])
                else:
                    self.w2i[word] = len(self.w2i.keys())  # add new word
                    wembeds.init_row(self.w2i[word], embeddings[word])
                init += 1
            print("initialized: {}".format(init), file=sys.stderr)

        else:
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim))
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim))

        # make it more flexible to add number of layers as specified by parameter
        layers = []  # inner layers
        # print("h_layers:", self.h_layers, file=sys.stderr)
        for layer_num in range(0, self.h_layers):
            # print(">>>", layer_num, "layer_num")

            if layer_num == 0:
                builder = dynet.LSTMBuilder(1, self.in_dim + self.c_in_dim * 2, self.h_dim,
                                            self.model)  # in_dim: size of each layer
                layers.append(BiRNNSequencePredictor(builder))  # returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                builder = dynet.LSTMBuilder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(builder))

                # store at which layer to predict task

        task_num_labels = len(self.tag2idx)
        output_layer = FFSequencePredictor(Layer(self.model, self.h_dim * 2, task_num_labels, dynet.softmax))

        char_rnn = RNNSequencePredictor(dynet.LSTMBuilder(1, self.c_in_dim, self.c_in_dim, self.model))

        predictors = {}
        predictors["inner"] = layers
        predictors["output_layers_dict"] = output_layer
        predictors["task_expected_at"] = self.h_layers

        return predictors, char_rnn, wembeds, cembeds

    def get_features(self, words):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if word in self.w2i:
                word_indices.append(self.w2i[word])
            else:
                word_indices.append(self.w2i["_UNK"])

            chars_of_word = [self.c2i["<w>"]]
            for char in word:
                if char in self.c2i:
                    chars_of_word.append(self.c2i[char])
                else:
                    chars_of_word.append(self.c2i["_UNK"])
            chars_of_word.append(self.c2i["</w>"])
            word_char_indices.append(chars_of_word)
        return word_indices, word_char_indices

    def get_data_as_indices(self, file_name):
        """
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [], []
        org_X, org_Y = [], []

        for (words, tags) in read_conll_file(file_name):
            word_indices, word_char_indices = self.get_features(words)
            tag_indices = [self.tag2idx.get(tag) for tag in tags]
            X.append((word_indices, word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
        return X, Y  # , org_X, org_Y - for now don't use

    def get_data_as_indices_from_instances(self, dev_words, dev_tags):
        """
        Extension of get_data_as_indices. Use words and tags rather than a file as input.
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [], []
        org_X, org_Y = [], []

        for (words, tags) in zip(dev_words, dev_tags):
            word_indices, word_char_indices = self.get_features(words)
            tag_indices = [self.tag2idx.get(tag) for tag in tags]
            X.append((word_indices, word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
        return X, Y  # , org_X, org_Y - for now don't use

    def predict(self, word_indices, char_indices, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        """
        dynet.renew_cg()  # new graph

        char_emb = []
        rev_char_emb = []
        # get representation for words
        for chars_of_token in char_indices:
            # use last state as word representation
            last_state = self.char_rnn.predict_sequence([self.cembeds[c] for c in chars_of_token])[-1]
            rev_last_state = self.char_rnn.predict_sequence([self.cembeds[c] for c in reversed(chars_of_token)])[-1]
            char_emb.append(last_state)
            rev_char_emb.append(rev_last_state)

        wfeatures = [self.wembeds[w] for w in word_indices]
        features = [dynet.concatenate([w, c, rev_c]) for w, c, rev_c in
                    zip(wfeatures, char_emb, reversed(rev_char_emb))]

        if train:  # only do at training time
            features = [dynet.noise(fe, self.noise_sigma) for fe in features]

        output_expected_at_layer = self.h_layers
        output_expected_at_layer -= 1

        # go through layers
        # input is now combination of w + char emb
        prev = features
        num_layers = self.h_layers
        for i in range(0, num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev)
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:
                output_predictor = self.predictors["output_layers_dict"]
                concat_layer = [dynet.concatenate([f, b]) for f, b in
                                zip(forward_sequence, reversed(backward_sequence))]

                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe, self.noise_sigma) for fe in concat_layer]
                output = output_predictor.predict_sequence(concat_layer)
                return output

            prev = forward_sequence
            prev_rev = backward_sequence  # not used

        raise Exception("oops should not be here")
        return None

    def evaluate(self, test_X, test_Y):
        """
        compute accuracy on a test file
        """
        correct = 0
        total = 0.0

        for i, ((word_indices, word_char_indices), gold_tag_indices) in enumerate(zip(test_X, test_Y)):
            output = self.predict(word_indices, word_char_indices)
            predicted_tag_indices = [np.argmax(o.value()) for o in output]

            correct += sum(
                [1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
            total += len(gold_tag_indices)

        return correct, total

    # Get train data: need to read each train set (linked to a task) separately

    def get_train_data(self, train_data):
        """
        transform training data to features (word indices)
        map tags to integers
        """
        X = []
        Y = []

        # word 2 indices and tag 2 indices
        w2i = {}  # word to index
        c2i = {}  # char to index
        tag2idx = {}  # tag2idx

        w2i["_UNK"] = 0  # unk word / OOV
        c2i["_UNK"] = 0  # unk char
        c2i["<w>"] = 1  # word start
        c2i["</w>"] = 2  # word end index

        num_sentences = 0
        num_tokens = 0
        for instance_idx, (words, tags) in enumerate(read_conll_file(train_data)):
            instance_word_indices = []  # sequence of word indices
            instance_char_indices = []  # sequence of char indices
            instance_tags_indices = []  # sequence of tag indices

            for i, (word, tag) in enumerate(zip(words, tags)):

                # map words and tags to indices
                if word not in w2i:
                    w2i[word] = len(w2i)
                instance_word_indices.append(w2i[word])

                chars_of_word = [c2i["<w>"]]
                for char in word:
                    if char not in c2i:
                        c2i[char] = len(c2i)
                    chars_of_word.append(c2i[char])
                chars_of_word.append(c2i["</w>"])
                instance_char_indices.append(chars_of_word)

                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

                instance_tags_indices.append(tag2idx.get(tag))

                num_tokens += 1

            num_sentences += 1

            X.append((instance_word_indices,
                      instance_char_indices))  # list of word indices, for every word list of char indices
            Y.append(instance_tags_indices)

        print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
        print("%s w features, %s c features " % (len(w2i), len(c2i)), file=sys.stderr)

        assert (len(X) == len(Y))

        # store mappings of words and tags to indices
        self.set_indices(w2i, c2i, tag2idx)

        return X, Y

    def get_train_data_from_instances(self, train_words, train_tags):
        """
        Extension of get_train_data method. Extracts training data from two arrays of word and label lists.
        transform training data to features (word indices)
        map tags to integers
        :param train_words: a numpy array containing lists of words
        :param train_tags: a numpy array containing lists of corresponding tags
        """
        X = []
        Y = []

        # word 2 indices and tag 2 indices
        w2i = {}  # word to index
        c2i = {}  # char to index
        tag2idx = {}  # tag2idx

        w2i["_UNK"] = 0  # unk word / OOV
        c2i["_UNK"] = 0  # unk char
        c2i["<w>"] = 1  # word start
        c2i["</w>"] = 2  # word end index

        num_sentences = 0
        num_tokens = 0
        for instance_idx, (words, tags) in enumerate(zip(train_words, train_tags)):
            instance_word_indices = []  # sequence of word indices
            instance_char_indices = []  # sequence of char indices
            instance_tags_indices = []  # sequence of tag indices

            for i, (word, tag) in enumerate(zip(words, tags)):

                # map words and tags to indices
                if word not in w2i:
                    w2i[word] = len(w2i)
                instance_word_indices.append(w2i[word])

                chars_of_word = [c2i["<w>"]]
                for char in word:
                    if char not in c2i:
                        c2i[char] = len(c2i)
                    chars_of_word.append(c2i[char])
                chars_of_word.append(c2i["</w>"])
                instance_char_indices.append(chars_of_word)

                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

                instance_tags_indices.append(tag2idx.get(tag))

                num_tokens += 1

            num_sentences += 1

            X.append((instance_word_indices,
                      instance_char_indices))  # list of word indices, for every word list of char indices
            Y.append(instance_tags_indices)

        print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
        print("%s w features, %s c features " % (len(w2i), len(c2i)), file=sys.stderr)

        assert (len(X) == len(Y))

        # store mappings of words and tags to indices
        self.set_indices(w2i, c2i, tag2idx)

        return X, Y


class MyNNTaggerArgumentOptions(object):
    def __init__(self):
        pass

    ### functions for checking arguments
    def acfunct(arg):
        """ check for allowed argument for --ac option """
        try:
            functions = [dynet.rectify, dynet.tanh]
            functions = {function.__name__: function for function in functions}
            functions["None"] = None
            return functions[str(arg)]
        except:
            raise argparse.ArgumentTypeError("String {} does not match required format".format(arg, ))


if __name__ == "__main__":
    main()
