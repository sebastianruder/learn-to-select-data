#!/bin/bash
#
# train models on UD 1.3
#
SUBMIT=0

PARTITION=nodes
mkdir -p runs

CORPUSDIR=~/corpora/pos/ud1.3/orgtok/goldpos/
EXPDIR=/data/p252438/experiments/bilty

tagger=bilty
mkdir -p $EXPDIR/models/$tagger
mkdir -p $EXPDIR/nohup
mkdir -p $EXPDIR/predictions/$tagger

ITERS=20    
#ITERS=30    
SIGMA=0.2
CDIM=100

SEED=1512141834
TRAINER=sgd
INDIM=64
HLAYERS=1
#HLAYERS=3
T0_OUT=$HLAYERS

for lang in `cat langs/lang_with_embeds.txt`; # all for which we have poly embeds (26)
do 
    TRAIN=$lang-ud-train.conllu
    JOBNAME=bilty-$lang-ud1.3-poly-i$ITERS-h$HLAYERS

    echo "#!/bin/bash"  > $$tmp
    echo "#SBATCH --ntasks=1 --cpus-per-task 12 --time=24:00:00 --job-name=$JOBNAME --partition=$PARTITION --mem=64GB" >> $$tmp
    echo "#SBATCH --output=runs/${JOBNAME}.out" >> $$tmp
    echo "#SBATCH --error=runs/${JOBNAME}.out2" >> $$tmp
    echo "module load CMake" >> $$tmp
    
    echo "python src/$tagger.py --dynet-seed $SEED --dynet-mem 1500 --train $CORPUSDIR/$TRAIN --test $CORPUSDIR/$lang-ud-test.conllu --dev $CORPUSDIR/$lang-ud-dev.conllu --output $EXPDIR/predictions/$tagger/$lang-ud-test.conllu.$JOBNAME --in_dim 64 --c_in_dim $CDIM --trainer $TRAINER --iters $ITERS --sigma $SIGMA --save $EXPDIR/models/$tagger/$JOBNAME.model --embeds embeds/poly_a/$lang.polyglot.txt --h_layers $HLAYERS --pred_layer $T0_OUT  > $EXPDIR/nohup/$JOBNAME.out 2> $EXPDIR/nohup/$tagger.$JOBNAME.out2" >> $$tmp

    if [ $SUBMIT -eq 1 ] ; then
	echo "SUBMIT"
        sbatch $$tmp
    fi
    cat $$tmp
    rm $$tmp
done

for lang in `cat langs/lang_canonic.txt` ;  # all without embeddings (but only canical names)
do 
    TRAIN=$lang-ud-train.conllu
    JOBNAME=bilty-$lang-ud1.3-i$ITERS-h$HLAYERS
    
    echo "#!/bin/bash"  > $$tmp
    echo "#SBATCH --ntasks=1 --cpus-per-task 12 --time=24:00:00 --job-name=$JOBNAME --partition=$PARTITION --mem=64GB" >> $$tmp
    echo "#SBATCH --output=runs/${JOBNAME}.out" >> $$tmp
    echo "#SBATCH --error=runs/${JOBNAME}.out2" >> $$tmp
    echo "module load CMake" >> $$tmp

    echo "python src/$tagger.py --dynet-seed $SEED --dynet-mem 1500 --train $CORPUSDIR/$TRAIN --test $CORPUSDIR/$lang-ud-test.conllu --dev $CORPUSDIR/$lang-ud-dev.conllu --output $EXPDIR/predictions/$tagger/$lang-ud-test.conllu.$JOBNAME --in_dim 64 --c_in_dim $CDIM --trainer $TRAINER --iters $ITERS --sigma $SIGMA --save $EXPDIR/models/$tagger/$JOBNAME.model --h_layers $HLAYERS --pred_layer $T0_OUT  > $EXPDIR/nohup/$JOBNAME.out 2> $EXPDIR/nohup/$tagger.$JOBNAME.out2" >> $$tmp

    if [ $SUBMIT -eq 1 ] ; then
        echo "SUBMIT"
        sbatch $$tmp
    fi

    cat $$tmp
    rm $$tmp
done
