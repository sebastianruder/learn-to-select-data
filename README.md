# Learning to select data for transfer learning with Bayesian Optimization

Sebastian Ruder, Barbara Plank (2017). [Learning to select data for transfer learning with Bayesian Optimization](https://arxiv.org/abs/1707.05246). _In Proceedings of the 2017 Conference 
on Empirical Methods in Natural Language Processing_, Copenhagen, Denmark.

## Requirements

### RoBO

The Robust Bayesian Optimization framework [RoBO](http://automl.github.io/RoBO/) needs to be installed.
It can be installed using the following steps:

1. First, install `libeigen3-dev` as a prerequisite:
`sudo apt-get install libeigen3-dev` (*)
2. Then, clone the RoBO repository: 
`git clone https://github.com/automl/RoBO.git`
3. Change into the directory: `cd RoBO/`
4. Install RoBOs requirements:
`for req in $(cat all_requirements.txt); do pip install $req; done`
5. Finally, install RoBO:
`python setup.py install`

For the topic models, `gensim` needs to be installed:
`pip install gensim`

### DyNet

We use the neural network library [DyNet](http://dynet.readthedocs.io/en/latest/index.html),
which works well with networks that have dynamic structures. DyNet can be 
installed by following the instructions [here](http://dynet.readthedocs.io/en/latest/python.html#manual-installation).

## Repository structure

- `bilstm_tagger`: The repository containing code for the Bi-LSTM tagger from 
Plank et al. (2016).
- `bist_parser`: The repository containing the code for the BIST parser from 
Kiperwasser and Goldberg (2016).
- `bayes_opt.py`: The main logic for running Bayesian Optimization.
- `constants.py`: Constants that are shared across all files.
- `data_utils.py`: Utility methods for data reading and processing.
- `features.py`: Methods for generating feature representations.
- `similarity.py`: Methods for measuring domain similarity.
- `simpletagger.py`: Code for running the Structured Perceptron POS tagger.
- `task_utils.py`: Utility methods for training and evaluation.

## Instructions

### Running Bayesian Optimization

The main logic for running Bayesian Optimization can be found in `bayes_opt.py`.
The features that are currently used are currently defined in `constants.py` as
`FEATURE_SETS` and are split into diversity and similarity features.
Bayesian Optimization minimizes the validation error on the specified dataset.

### Example usage

```
python bayes_opt.py --dynet-autobatch 1 -d data/gweb_sancl -m models/model \
                    -t emails newsgroups reviews weblogs wsj --task pos \
                    -b random most-similar-examples \
                    --parser-output-path parser_outputs \
                    --perl-script-path bist_parser/bmstparser/src/util_scripts/eval.pl \
                    -f similarity --z-norm --num-iterations 100 \
                    --num-runs 1 --log-file logs/log
```

- `dynet-autobatch 1`: use DyNet auto-batching
- `-d data/gweb_sancl`: use the data from the SANCL 2012 shared task
- `-m models/model`: specify the directory where the model should be saved
- `-t emails newsgroups reviews weblogs wsj`: adapt to the specified target 
domains in the order they were provided
- `--task pos`: perform POS tagging with the Structured Perceptron model
- `-b`: use the random and most-similar-examples baselines
- `--parser-output-path`, `--perl-script-path`: only required when performing 
parsing
- `-f`: use only similarity features with Bayesian Optimization
- `--z-norm`: perform z-normalisation (recommended)
- `--num-iterations`: perform 100 iterations of Bayesian Optimization
- `--num-runs`: perform one run of Bayesian Optimization per target domain
- `--log-file`: log the results of the baselines and Bayesian Optimization to
 this file

### Adding a new task

In order to add a new task, you need to do several things:
- Add the new task to `TASKS`, `TASK2TRAIN_EXAMPLES`, and `TASK2DOMAINS` in 
`constants.py`.
- Add a method to read data for the task to `data_utils.py` and add the 
mapping to `data_utils.task2read_data_func`.
- Add a method to train and evaluate the task to `task_utils.py` and add the 
mapping to `task_utils.task2train_and_evaluate_func`.
- Add the function that should be minimized to `bayes_opt.py` and add the 
mapping to `task2_objective_function`. The function should take
as input the feature weights and output the error.

### Adding new features

New feature sets or features can be added by adding them to `constants.py`.
Similarity features or new representations can be added to 
`similarity.py`. Diversity features or any other features can to be added to
`features.py`. All new features must be added to 
`get_feature_representations` and `get_feature_names` in `features.py`.



## Data

### Multi-Domain Sentiment Dataset

The Amazon Reviews Multi-Domain Sentiment Dataset (Blitzer et al., 2007)
used in the current Bayesian Optimization experiment can be downloaded
using the following steps:
1. Create a new `amazon-reviews` directory:
`mkdir amazon-reviews`
2. Change into the directory:
`cd amazon-reviews`
3. Download the dataset:
`wget https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz`
4. Extract the dataset:
`tar -xvf processed_acl.tar.gz`

In `bayes_opt.py`, the `data-path` argument should now be pointed to
the `amazon-reviews` directory.

### Multi-domain POS and parsing data

We use the data from the [SANCL 2012 shared task/English Web Treebank](https://catalog.ldc.upenn.edu/ldc2012t13).

### Word embedding data

Pre-trained word embeddings can be downloaded from [here](http://nlp.stanford.edu/projects/glove/).
We are using GloVe embeddings in the paper, but other pre-trained embeddings are also possible.
Smaller embedding files can be used for faster iteration.


## Models

### BIST parser

We use the BIST parser from Kiperwasser and Goldberg (2016) for our experiments. The parser repo can be found
[here](https://github.com/elikip/bist-parser) and was integrated using [`git submodule`](http://stackoverflow.com/questions/2140985/how-to-set-up-a-git-project-to-use-an-external-repo-submodule).

For running the parser with Bayesian Optimization, two additional hyperparameters are necessary:
- `--perl-script-path`: This is the location of the `perl` script that is used to evaluate the parser's predictions.
                        The script is located in `bist_parser/bmstparser/src/util_scripts/eval.pl` per default.
- `--parser-output-path`: This is the location of the folder where the parser's predictions and the output of the
                          `perl` script will be written to.

Per default, Labeled Attachment Score on the held-out validation set is used to evaluate the parser's performance and 
evaluation results are saved to a subfolders of `parser-output-path` that indicate the target domain and feature sets
used. Another subsubfolder is created for the best weights configuration so that Labeled Attachment Score, Unlabeled
Attachment Score and Accuracy as well as other statistics are available for the final test set evaluation.

### Bi-LSTM tagger

The Bi-LSTM tagger we are using is a simplified, single-task version of the
hierarchical Multi-task Bi-LSTM tagger used by Plank et al. (2016). The source
repository of the tagger can be found [here](https://github.com/bplank/bilstm-aux/).

## (*) Installing Eigen without sudo rights

In case you you do not have sudo rights to run `sudo apt-get install
libeigen3-dev` here is a workaround.

Create a folder where you download the sources of libeigen3-dev:

```
mkdir -p tools/eigen3
cd tools/eigen3
apt-get source libeigen3-dev
```

Afterwards point the required packages for `RoBo` to the folder just created: `tools/eigen3/eigen3-3.2.0`

For instance, to install the 'george' requirement of `RoBo`, add the `--global-option` parameters pointing to the eigen directory:

```
pip install git+https://github.com/sfalkner/george.git --global-option=build_ext --global-option=-I/path/to/tools/eigen3/eigen3-3.2.0
```

(see http://dan.iel.fm/george/current/user/quickstart/#installation -> if you have Eigen in a strange place)


## Reference

If you make use of the contents of this repository, we appreciate citing the following paper:
```
@inproceedings{ruder2017select,
  title={{Learning to select data for transfer learning with Bayesian Optimization}},
  author={Ruder, Sebastian and Plank, Barbara},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year={2017}
}
```

