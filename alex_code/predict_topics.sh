#!/bin/sh
# Outputs Topics Proportions for file based on trained LDA model
# bash predict_topics.sh <mallet_results_dir> <text file> <output file>
cd mallet-2.0.8
bin/mallet import-file --input $2 --output $3 --keep-sequence TRUE --remove-stopwords --token-regex '[\p{L}\p{M}]+' --use-pipe-from $1/text.mallet
bin/mallet infer-topics --inferencer $1/inferencer --input $3 --output-doc-topics $3

