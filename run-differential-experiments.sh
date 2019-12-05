#!/bin/bash

NUM_REPETITIONS=10

#cat /dev/null > results/cf-1.csv
#for n in $(seq 1 $NUM_REPETITIONS); do
#	cargo run --release --bin differential_itembased_experiments -- -w1 100 >> results/cf-1.csv;
#done;
#
#cat /dev/null > results/cf-2.csv
#for n in $(seq 1 $NUM_REPETITIONS); do
#	cargo run --release --bin differential_itembased_experiments -- -w2 100 >> results/cf-2.csv;
#done;

cat /dev/null > results/cf-3.csv
for n in $(seq 1 $NUM_REPETITIONS); do
	cargo run --release --bin differential_itembased_experiments -- -w3 100 >> results/cf-3.csv;
done;

#cat /dev/null > results/cf-4.csv
#for n in $(seq 1 $NUM_REPETITIONS); do
#	cargo run --release --bin differential_itembased_experiments -- -w4 100 >> results/cf-4.csv;
#done;


