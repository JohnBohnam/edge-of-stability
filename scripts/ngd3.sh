#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR"/environ.sh
time $PYTHON "$SCRIPT_DIR"/../src/run_ngd.py \
	cifar10 \
	cnn-relu \
	mse \
	0.01 \
	10000 \
	--acc_goal 0.99 \
	--neigs 2 \
	--eig_freq 20 \
	--iterate_freq 50 \
	--save_model true \
	--physical_batch_size 1000 \
	--save_freq 50 \
	--epsilon 1e-6 \
	--momentum 0.5 \