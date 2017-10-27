#!/bin/sh
#run_mallet.sh <text_corpus> <mallet_results> <num_topics>

cd 'mallet-2.0.8'
./bin/mallet import-dir --input $1 --output $2/text.mallet --keep-sequence TRUE --remove-stopwords --skip-html --extra-stopwords sina_stopwords.txt --token-regex '[\p{L}\p{M}]+'
./bin/mallet train-topics --input $2/text.mallet --output-model $2/mallet_model.mm --output-state $2/state.txt.gz --output-topic-keys $2/keys.txt --xml-topic-phrase-report $2/phrases.xml --output-doc-topics $2/doc_topics.txt --num-threads 4 --num-topics $3 --num-top-words 200 --inferencer-filename $2/inferencer
cd ..

python2 HannahWallach/src/summarize.py --state $2/state.txt.gz --topic-keys $2/keys.txt --dist empirical --test chi-squared-yates --selection n-1-gram > $2/terms.txt

