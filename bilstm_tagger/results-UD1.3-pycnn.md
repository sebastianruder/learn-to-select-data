
#### Results on UD1.3

NB. The results below are with the old version of Dynet (pycnn).

The table below provides results on UD1.3 (iters=20, h_layers=1).

+poly is using pre-trained embeddings to initialize
word embeddings.  Note that for some languages it slightly hurts performance.

```
python src/bilty.py --dynet-seed 1512141834 --dynet-mem 1500 --train /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-train.conllu --test /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-test.conllu --dev /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-dev.conllu --output /data/$user/experiments/bilty/predictions/bilty/en-ud-test.conllu.bilty-en-ud1.3-poly-i20-h1 --in_dim 64 --c_in_dim 100 --trainer sgd --iters 20 --sigma 0.2 --save /data/$user/experiments/bilty/models/bilty/bilty-en-ud1.3-poly-i20-h1.model --embeds embeds/poly_a/en.polyglot.txt --h_layers 1 --pred_layer 1  > /data/$user/experiments/bilty/nohup/bilty-en-ud1.3-poly-i20-h1.out 2> /data/$user/experiments/bilty/nohup/bilty.bilty-en-ud1.3-poly-i20-h1.out2
```

| Lang | i20-h1  | +poly |
| ---| -----:| -----:|
| ar | 96.07 | 96.37 |
| bg | 98.21 | 98.12 |
| ca | 98.11 | 98.24 |
| cs | 98.63 | 98.60 |
| cu | 96.48 | -- |
| da | 96.06 | 96.04 |
| de | 92.91 | 93.64 |
| el | 97.85 | 98.36 |
| en | 94.60 | 95.04 |
| es | 95.23 | 95.76 |
| et | 95.75 | 96.57 |
| eu | 93.86 | 95.40 |
| fa | 96.82 | 97.38 |
| fi | 94.32 | 95.35 |
| fr | 96.34 | 96.45 |
| ga | 90.50 | 91.29 |
| gl | 96.89 | -- |
| got | 95.97 | -- |
| grc | 94.36 | -- |
| he | 95.25 | 96.78 |
| hi | 96.37 | 96.93 |
| hr | 94.98 | 96.07 |
| hu | 93.84 | -- |
| id | 93.17 | 93.55 |
| it | 97.40 | 97.82 |
| kk | 77.68 | -- |
| la | 90.17 | -- |
| lv | 91.42 | -- |
| nl | 90.02 | 89.87 |
| no | 97.58 | 97.97 |
| pl | 96.30 | 97.36 |
| pt | 97.21 | 97.46 |
| ro | 95.49 | -- |
| ru | 95.69 | -- |
| sl | 97.53 | 96.42 |
| sv | 96.49 | 96.76 |
| ta | 84.51 | -- |
| tr | 93.81 | -- |
| zh | 93.13 | -- |

Using pre-trained embeddings often helps to improve accuracy, however, does not
strictly hold for all languages.

For more information, predictions files and pre-trained models
visit [http://www.let.rug.nl/bplank/bilty/](http://www.let.rug.nl/bplank/bilty/)



