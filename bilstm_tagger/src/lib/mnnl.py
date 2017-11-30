"""
my NN library
(based on Yoav's)
"""
from itertools import count
import dynet
import numpy as np

import sys

global_counter = count(0)

## NN classes
class SequencePredictor:
    def __init__(self):
        pass
    
    def predict_sequence(self, inputs):
        raise NotImplementedError("SequencePredictor predict_sequence: Not Implmented")

class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict_sequence(self, inputs):
        return [self.network_builder(x) for x in inputs]

class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder or SimpleRNNBuilder object.
        """
        self.builder = rnn_builder
        
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return [x.output() for x in s_init.add_inputs(inputs)] #quicker version

class BiRNNSequencePredictor(SequencePredictor):

    def __init__(self, lstm_builder):
        # use single one
        self.builder = lstm_builder
    def predict_sequence(self, inputs):
        f_init = self.builder.initial_state()
        b_init = self.builder.initial_state()
        forward_sequence = [x.output() for x in f_init.add_inputs(inputs)]
        backward_sequence = [x.output() for x in b_init.add_inputs(reversed(inputs))]
        return forward_sequence, backward_sequence  # do concat only later! return separate forward and backward seq
        
class Layer:
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh):
        ident = str(next(global_counter))
        self.act = activation
        self.W = model.add_parameters((output_dim, in_dim)) 
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        W = dynet.parameter(self.W)
        b = dynet.parameter(self.b)
        return self.act(W*x + b)
