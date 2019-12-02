#!/bin/bash

NUM_REPETITIONS=20

cat /dev/null > results/mnb-1.csv
for n in $(seq 1 $NUM_REPETITIONS); do
	cargo run --release --bin differential_mnb_experiments -- -w1 25 >> results/mnb-1.csv;
done;

cat /dev/null > results/mnb-2.csv
for n in $(seq 1 $NUM_REPETITIONS); do
	cargo run --release --bin differential_mnb_experiments -- -w2 50 >> results/mnb-2.csv;
done;

cat /dev/null > results/mnb-4.csv
for n in $(seq 1 $NUM_REPETITIONS); do
	cargo run --release --bin differential_mnb_experiments -- -w4 100 >> results/mnb-4.csv;
done;

