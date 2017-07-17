"""
Utility methods that are used for training and evaluation of the tasks.
"""

import os
import operator
import numpy as np
import random
from collections import namedtuple

from sklearn import svm
from sklearn.metrics import accuracy_score

import data_utils
from constants import POS_ID, NEG_ID, SENTIMENT, POS, POS_BILSTM, PARSING,\
    BAYES_OPT
from simpletagger import StructuredPerceptron

from bist_parser.bmstparser.src import mstlstm
from bist_parser.bmstparser.src.utils import vocab_conll, write_conll,\
    write_original_conll

from bilstm_tagger.src.simplebilty import SimpleBiltyTagger, load

NUM_EPOCHS = 50
PATIENCE = 2


def get_data_subsets(feature_vals, feature_weights, train_data, train_labels,
                     task, num_train_examples):
    """
    Given the feature values and the feature weights, return the stratified
    subset of the training data with the highest feature scores.
    :param feature_vals: a numpy array of shape (num_train_data, num_features)
                         containing the feature values
    :param feature_weights: a numpy array of shape (num_features, ) containing
                            the weight for each feature
    :param train_data: a sparse numpy array of shape (num_train_data, vocab_size)
                       containing the training data
    :param train_labels: a numpy array of shape (num_train_data) containing the
                         training labels
    :param task: the task; this determines whether we use stratification
    :param num_train_examples: the number of training examples for the
                               respective task
    :return: subsets of the training data and its labels as a tuple of two
             numpy arrays
    """
    # calculate the scores as the dot product between feature values and weights
    scores = feature_vals.dot(np.transpose(feature_weights))

    # sort the indices by their scores
    sorted_index_score_pairs = sorted(zip(range(len(scores)), scores),
                                      key=operator.itemgetter(1), reverse=True)

    # get the top indices
    top_indices, _ = zip(*sorted_index_score_pairs)

    if task == SENTIMENT:
        # for sentiment, rather than taking the top n indices, we still want to
        # have a stratified training set so we take the top n/2 positive and
        # top n/2 negative indices
        top_pos_indices = [idx for idx in top_indices if train_labels[idx] ==
                           POS_ID][:int(num_train_examples/2)]
        top_neg_indices = [idx for idx in top_indices if train_labels[idx] ==
                           NEG_ID][:int(num_train_examples/2)]
        top_indices = top_pos_indices + top_neg_indices
    elif task in [POS, POS_BILSTM, PARSING]:
        # for POS tagging and parsing, we don't need a stratified train set
        top_indices = list(top_indices[:num_train_examples])
    else:
        raise ValueError('Top index retrieval not implemented for %s.' % task)

    if isinstance(train_data, list):
        # numpy indexing does not work if train_data is a list
        return [train_data[idx] for idx in top_indices],\
               train_labels[top_indices]

    # we get the corresponding subsets of the training data and the labels
    return train_data[top_indices], train_labels[top_indices]


def task2train_and_evaluate_func(task):
    """Return the train_and_evaluate function for a task."""
    if task == SENTIMENT:
        return train_and_evaluate_sentiment
    if task == POS:
        return train_and_evaluate_pos
    if task == POS_BILSTM:
        return train_and_evaluate_pos_bilstm
    if task == PARSING:
        return train_and_evaluate_parsing
    raise ValueError('Train_and_evaluate is not implemented for %s.' % task)


def train_and_evaluate_sentiment(train_data, train_labels, val_data, val_labels,
                                 test_data=None, test_labels=None,
                                 parser_output_path=None, perl_script_path=None):
    """
    Trains an SVM on the provided training data. Calculates accuracy on the
    validation set and (optionally) on the test set.
    :param train_data: the training data; a sparse numpy matrix of shape
                       (num_examples, max_vocab_size)
    :param train_labels: the training labels; a numpy array of shape (num_labels)
    :param val_data: the validation data; same format as the training data
    :param val_labels: the validation labels
    :param test_data: the test data
    :param test_labels: the test labels
    :param parser_output_path: only necessary for parsing; is ignored here
    :param perl_script_path: only necessary for parsing; is ignored here
    :return: the validation accuracy and (optionally) the test data;
            otherwise None
    """
    print('Training the SVM on %d examples...' % train_data.shape[0])
    clf = svm.SVC()
    clf.fit(train_data, train_labels)

    # validate the configuration on the validation and test set (if provided)
    val_predictions = clf.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print('Val acc: %.5f' % val_accuracy)
    test_accuracy = None
    if test_data is not None and test_labels is not None:
        test_predictions = clf.predict(test_data)
        test_accuracy = accuracy_score(test_labels, test_predictions)
        print('Test acc: %.5f' % test_accuracy)
    return val_accuracy, test_accuracy


def train_and_evaluate_pos(train_data, train_labels, val_data, val_labels,
                           test_data=None, test_labels=None,
                           parser_output_path=None, perl_script_path=None):
    """
    Trains the tagger on the provided training data. Calculates accuracy on the
    validation set and (optionally) on the test set.
    :param train_data: the training data; a list of lists of shape
                       (num_examples, sequence_length)
    :param train_labels: the training labels; a list of lists of tags
    :param val_data: the validation data; same format as the training data
    :param val_labels: the validation labels
    :param test_data: the test data
    :param test_labels: the test labels
    :param parser_output_path: only necessary for parsing; is ignored here
    :param perl_script_path: only necessary for parsing; is ignored here
    :return: the validation accuracy and (optionally) the test acc; else None
    """
    print('Training the tagger on %d examples...' % len(train_data))
    sp = StructuredPerceptron()
    tr_data = [(words, tags) for words, tags in zip(train_data, train_labels)]
    pos_iterations, pos_learning_rate = 5, 0.2
    sp.fit(tr_data, iterations=pos_iterations, learning_rate=pos_learning_rate)

    # validate the configuration on the validation and test set (if provided)
    val_predictions = sp.predict(val_data)

    val_accuracy = pos_accuracy_score(val_labels, val_predictions)
    print('Val acc: %.5f' % val_accuracy)

    test_accuracy = None
    if test_data is not None and test_labels is not None:
        test_predictions = sp.predict(test_data)
        test_accuracy = pos_accuracy_score(test_labels, test_predictions)
        print('Test acc: %.5f' % test_accuracy)
    return val_accuracy, test_accuracy


def train_and_evaluate_pos_bilstm(train_data, train_labels, val_data, val_labels,
                                  test_data=None, test_labels=None,
                                  parser_output_path=None, perl_script_path=None):
    """
    Trains the tagger on the provided training data. Calculates accuracy on the
    validation set and (optionally) on the test set.
    :param train_data: the training data; a list of lists of shape
                       (num_examples, sequence_length)
    :param train_labels: the training labels; a list of lists of tags
    :param val_data: the validation data; same format as the training data
    :param val_labels: the validation labels
    :param test_data: the test data
    :param test_labels: the test labels
    :return: the validation accuracy and (optionally) the test data; else None
    """
    print('Training the BiLSTM tagger on %d examples...' % len(train_data))
    in_dim = 64
    h_dim = 100
    c_in_dim = 100
    h_layers = 1
    trainer = "adam"
    # temporary file used to restore best model; random number is used to avoid
    # name clash in parallel runs
    model_path = '/tmp/bilstm_tagger_model_%d' % random.randint(0, 1000000)
    tagger = SimpleBiltyTagger(in_dim, h_dim, c_in_dim, h_layers,
                               embeds_file=None)
    train_X, train_Y = tagger.get_train_data_from_instances(train_data,
                                                            train_labels)
    val_X, val_Y = tagger.get_data_as_indices_from_instances(val_data,
                                                             val_labels)

    # train the model with early stopping
    tagger.fit(train_X, train_Y, NUM_EPOCHS, trainer, val_X=val_X, val_Y=val_Y,
               patience=PATIENCE, model_path=model_path)

    # load the best model and remove the model files
    tagger = load(model_path)
    os.unlink(model_path)
    os.unlink(model_path + '.pickle')  # file used to save the parameters
    val_correct, val_total = tagger.evaluate(val_X, val_Y)
    val_accuracy = val_correct / val_total
    print('Val acc: %.5f' % val_accuracy)

    test_accuracy = None
    if test_data is not None and test_labels is not None:
        test_X, test_Y = tagger.get_data_as_indices_from_instances(test_data,
                                                                   test_labels)
        test_correct, test_total = tagger.evaluate(test_X, test_Y)
        test_accuracy = test_correct / test_total
        print('Test acc: %.5f' % test_accuracy)
    return val_accuracy, test_accuracy


def train_and_evaluate_parsing(train_data, train_labels, val_data, val_labels,
                               test_data=None, test_labels=None,
                               parser_output_path=None, perl_script_path=None):
    """
    Trains the parser on the provided training data. Calculates LAS on the
    validation set and (optionally) on the test set.
    :param train_data: the training data; a list of CoNLL entries
    :param train_labels: pseudo-labels; not used as labels as labels are
                         contained in train_data
    :param val_data: the validation data; same format as the training data
    :param val_labels: pseud-labels; not used as contained in val_data
    :param test_data: the test data
    :param test_labels: pseudo-labels; not used as contained in test_data
    :return: the validation accuracy and (optionally) the test data; else None
    """
    print('Training the parser on %d examples...' % len(train_data))
    if test_data is not None:
        # incorporate the test data as some POS tags (e.g. XX) might only
        # appear in the target domain
        words, w2i, pos, rels = vocab_conll(np.hstack([train_data, val_data, test_data]))
    else:
        words, w2i, pos, rels = vocab_conll(np.hstack([train_data, val_data]))

    # set the variables used for initializing the parser and initialize the
    # parser
    ParserOptions = namedtuple('parser_options',
                               'activation, blstmFlag, labelsFlag, costaugFlag,'
                               ' bibiFlag, lstm_dims, wembedding_dims, '
                               'pembedding_dims, rembedding_dims, lstm_layers, '
                               'external_embedding, hidden_units, '
                               'hidden2_units, epochs')
    parser_options = ParserOptions(
        epochs=NUM_EPOCHS,
        activation='tanh',
        blstmFlag=True,
        labelsFlag=True,
        costaugFlag=True,
        bibiFlag=False,
        lstm_dims=125,
        wembedding_dims=100,
        pembedding_dims=25,
        rembedding_dims=25,
        lstm_layers=2,
        external_embedding=None,
        hidden_units=100,
        hidden2_units=0
    )
    parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, parser_options)

    # write the dev data to a file
    dev_data_path = os.path.join(parser_output_path, 'dev.conll')
    write_original_conll(dev_data_path, val_data)

    # set the variables used for tracking training progress for early stopping
    best_dev_las, epochs_no_improvement = 0., 0
    best_model_path = os.path.join(parser_output_path, 'parser')
    print('Training model for %d max epochs with early stopping with patience '
          '%d...' % (NUM_EPOCHS, PATIENCE))
    for epoch in range(parser_options.epochs):
        print('Starting epoch', epoch)
        parser.TrainOnEntries(train_data)

        # write the predictions to a file
        pred_path = os.path.join(parser_output_path,
                                 'dev_pred_epoch_' + str(epoch + 1) + '.conll')
        write_conll(pred_path, parser.PredictOnEntries(val_data))
        eval_path = pred_path + '.eval'
        perl_script_command = ('perl %s -g %s -s %s > %s' % (
            perl_script_path,dev_data_path, pred_path, eval_path))
        print('Evaluating with %s...' % perl_script_command)
        os.system(perl_script_command)
        las, uas, acc = data_utils.read_parsing_evaluation(eval_path)

        # remove the predictions and the evaluation file
        if os.path.exists(pred_path):
            os.unlink(pred_path)
        if os.path.exists(eval_path):
            os.unlink(eval_path)
        if las > best_dev_las:
            print('LAS %.2f is better than best dev LAS %.2f.'
                  % (las, best_dev_las))
            best_dev_las = las
            epochs_no_improvement = 0
            parser.Save(best_model_path)
        else:
            print('LAS %.2f is worse than best dev LAS %.2f.'
                  % (las, best_dev_las))
            epochs_no_improvement += 1
        if epochs_no_improvement == PATIENCE:
            print('No improvement for %d epochs. Early stopping...'
                  % epochs_no_improvement)
            print('Best dev LAS:', best_dev_las)
            break

    test_las = None
    if test_data is not None:
        # load the best model
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, parser_options)
        parser.Load(best_model_path)

        # first write the dev data to a file
        test_data_path = os.path.join(parser_output_path, 'test.conll')
        write_original_conll(test_data_path, test_data)

        # then write the prediction to another file
        pred_path = os.path.join(parser_output_path, 'test_pred.conll')
        write_conll(pred_path, parser.PredictOnEntries(test_data))
        eval_path = pred_path + '.eval'
        perl_script_command = ('perl %s -g %s -s %s > %s' % (
            perl_script_path, test_data_path, pred_path, eval_path))
        print('Evaluating with %s...' % perl_script_command)
        os.system(perl_script_command)
        test_las, test_uas, test_acc = data_utils.read_parsing_evaluation(
            eval_path)
        print('Test LAS:', test_las, 'test UAS:', test_uas,
              'test acc:', test_acc)

    # remove the saved parser
    if os.path.exists(best_model_path):
        os.unlink(best_model_path)
    return best_dev_las, test_las


def train_pretrained_weights(feature_values, X_train, y_train, train_domains,
                             num_train_examples, X_val, y_val, X_test, y_test,
                             trg_domain, args, feature_names,
                             parser_output_path, perl_script_path):
    """
    Train a model using pre-trained data selection weights (which could have
    been trained on an other model/domain/task).
    :param feature_values: a numpy array of shape (num_examples, num_features)
    :param X_train: the training data
    :param y_train: the training labels
    :param train_domains: a list of training domains, only used for counting
    :param num_train_examples: the number of examples used for training
    :param X_val: the validation data
    :param y_val: the validation labels
    :param X_test: the test data
    :param y_test: the test labels
    :param trg_domain: the target domain
    :param args: the arguments used for calling the script; used for logging
    :param feature_names: a list of the feature names
    :param parser_output_path: the output path of the parser
    :param perl_script_path: the path to the perl script
    :return:
    """
    for feat_weights_domain, feat_weights_feats, feature_weights in \
            data_utils.read_feature_weights_file(args.feature_weights_file):
        assert len(feature_weights) == len(feature_names)
        assert set(args.feature_sets) == set(feat_weights_feats.split(' '))

        if trg_domain != feat_weights_domain:
            continue

        # count how many examples belong to each source domain
        train_domain_subset, _ = get_data_subsets(
            feature_values, feature_weights, train_domains, y_train, args.task,
            num_train_examples)
        for subset_domain in set(train_domain_subset):
            print('# of %s in train data for trg domain %s: %d'
                  % (subset_domain, trg_domain,
                     train_domain_subset.count(subset_domain)))
            continue

        # get the train subset with the highest scores and train
        train_subset, labels_subset = get_data_subsets(
            feature_values, feature_weights, X_train, y_train, args.task,
            num_train_examples)
        val_accuracy, test_accuracy = task2train_and_evaluate_func(args.task)(
            train_subset, labels_subset, X_val, y_val, X_test, y_test,
            parser_output_path=parser_output_path,
            perl_script_path=perl_script_path)
        dict_key = ('%s-X-domain-%s-%s' % (BAYES_OPT, feat_weights_domain,
                                           feat_weights_feats))

        # log the result to the log file
        data_utils.log_to_file(args.log_file, {dict_key: [(
            val_accuracy, test_accuracy, feature_weights)]}, trg_domain, args)


def pos_accuracy_score(gold, predicted):
    """
    Calculate the accuracy for POS.
    :param gold: a list of lists of gold tags
    :param predicted: a list of lists of predicted tags
    :return the accuracy score
    """
    tags_correct = np.sum([1 for gold_tags, pred_tags in zip(gold, predicted)
                           for g, p in zip(gold_tags, pred_tags) if g == p])
    tags_total = len([t for g in gold for t in g])  # ravel list
    return tags_correct/float(tags_total)
