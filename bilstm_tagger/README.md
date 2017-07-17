## bi-LSTM tagger

Bidirectional Long-Short Term Memory tagger 

If you use this tagger please cite our paper:
http://arxiv.org/abs/1604.05529

### Requirements

* python3 
* [dynet](https://github.com/clab/dynet)

## Installation

Download and install dynet in a directory of your choice DYNETDIR: 

```
mkdir $DYNETDIR
git clone https://github.com/clab/dynet
```

Follow the instructions in the Dynet documentation (use `-DPYTHON`,
see http://dynet.readthedocs.io/en/latest/python.html). 

And compile dynet:

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python`
```

(if you have a GPU:

```
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python` -DBACKEND=cuda
```
)

After successful installation open python and import dynet, you can
test if the installation worked with:

```
>>> import dynet
[dynet] random seed: 2809331847
[dynet] allocating memory: 512MB
[dynet] memory allocation done.
>>> dynet.__version__
2.0
```

(You may need to set you PYTHONPATH to include Dynet's `build/python`)

#### DyNet supports python 3

The old bilstm-aux had a patch to work with python 3. This
is no longer necessary, as DyNet supports python 3 as of
https://github.com/clab/dynet/pull/130#issuecomment-259656695


#### Example command

Training the tagger:

```
python src/bilty.py --dynet-seed 1512141834 --dynet-mem 1500 --train /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-train.conllu --test /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-test.conllu --dev /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-dev.conllu --output /data/$user/experiments/bilty/predictions/bilty/en-ud-test.conllu.bilty-en-ud1.3-poly-i20-h1 --in_dim 64 --c_in_dim 100 --trainer sgd --iters 20 --sigma 0.2 --save /data/$user/experiments/bilty/models/bilty/bilty-en-ud1.3-poly-i20-h1.model --embeds embeds/poly_a/en.polyglot.txt --h_layers 1 --pred_layer 1  > /data/$user/experiments/bilty/nohup/bilty-en-ud1.3-poly-i20-h1.out 2> /data/$user/experiments/bilty/nohup/bilty.bilty-en-ud1.3-poly-i20-h1.out2
```

#### Embeddings

The poly embeddings [(Al-Rfou et al.,
2013)](https://sites.google.com/site/rmyeid/projects/polyglot) can be
downloaded from [here](http://www.let.rug.nl/bplank/bilty/embeds.tar.gz) (0.6GB)


#### A couple of remarks

The choice of 22 languages from UD1.2 (rather than 33) is described in
our TACL parsing paper, Section 3.1. [(Agić et al.,
2016)](https://transacl.org/ojs/index.php/tacl/article/view/869). Note,
however, that the bi-LSTM tagger does not require large amounts of
training data (as discussed in our paper). Therefore above are 
results for all languages in UD1.3 (for the canonical language
subparts, i.e., those with just the language prefix, no further
suffix; e.g. 'nl' but not 'nl_lassy', and those languages which are
distributed with word forms).

The `bilty` code is a significantly refactored version of the code
originally used in the paper. For example, `bilty` supports multi-task
learning with output layers at different layers (`--pred_layer`), and
it correctly supports stacked LSTMs (see e.g., Ballesteros et al.,
2015, Dyer et al., 2015). The results on UD1.3 are obtained with
`bilty` using no stacking (`--h_layers 1`). 

#### Recommended setting for `bilty`:

* 3 stacked LSTMs, predicting on outermost layer, otherwise default settings, i.e., `--h_layers 3 --pred_layer 3`

#### Reference

```
@inproceedings{plank:ea:2016,
  title={{Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss}},
  author={Plank, Barbara and S{\o}gaard, Anders and Goldberg, Yoav},
  booktitle={ACL 2016, arXiv preprint arXiv:1604.05529},
  url={http://arxiv.org/abs/1604.05529},
  year={2016}
}
```

