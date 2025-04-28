#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR"/environ.sh
$PYTHON "$SCRIPT_DIR"/../src/flow.py \
	cifar10-5k \
	fc-tanh \
	mse \
	1.0 \
	1000 \
	--acc_goal 0.99 \
	--neigs 2 \
	--eig_freq 100 \
	--iterate_freq 50 \
	--save_model true \
	--save_freq 50
