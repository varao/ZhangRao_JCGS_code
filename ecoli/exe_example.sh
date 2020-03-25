#!/bin/sh -l
sampleSize=${1:-2000};
covSampleSize=${2:-2000};


if [ ! -d results ]; then
	mkdir results;
fi
python simulation.py $sampleSize $covSampleSize
