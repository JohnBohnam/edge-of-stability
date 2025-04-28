#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR"/environ.sh
time $PYTHON "$SCRIPT_DIR"/../src/gd.py \
	cifar10 \
	fc-tanh \
	mse \
	0.05 \
	100 \
	--acc_goal 0.99 \
	--neigs 2 \
	--eig_freq 20 \
	--iterate_freq 50 \
	--save_model true \
	--physical_batch_size 5000 \
	--save_freq 50
