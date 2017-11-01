#!/bin/sh

declare -a arr=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)

rm -rf results_json_diftopics/* ####

for i in "${arr[@]}"
do
	echo $i
	python run_everything.py $i > out.log 2> err.log && cp -r results_json results_json_diftopics/$i > out.log 2> err.log 
done

