#### Example of using bilty from within code
## 
## to properly seed dyNet add parameter to your script:
## python run_simply.py --dynet-seed 113

from bilstm_tagger.src.simplebilty import SimpleBiltyTagger
import random
### Use --dynet-seed $SEED
seed=113 # assume we pass this to script
train_data = "/Users/bplank/corpora/pos/ud1.3/orgtok/goldpos/da-ud-dev.conllu"
dev_data = "/Users/bplank/corpora/pos/ud1.3/orgtok/goldpos/da-ud-test.conllu"
in_dim=64
h_dim=100
c_in_dim=100
h_layers=1
iters=2
trainer="sgd"
tagger = SimpleBiltyTagger(in_dim, h_dim,c_in_dim,h_layers,embeds_file=None)
train_X, train_Y = tagger.get_train_data(train_data)
tagger.fit(train_X, train_Y, iters, trainer,seed=seed)
test_X, test_Y = tagger.get_data_as_indices(dev_data)
correct, total = tagger.evaluate(test_X, test_Y)
print(correct, total, correct/total)
