#!/bin/sh -l
dm=${1:-5};
alpha=${2:-2};
beta=${3:-2};
sampleSize=${4:-5000};
tend=${5:-10};
if [ ! -d results ]; then
	mkdir results;
fi
python simulation.py $dm $alpha $beta $sampleSize $tend
