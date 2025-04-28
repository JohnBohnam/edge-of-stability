#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/environ.sh"
$PYTHON "$SCRIPT_DIR"/../src/adam.py \
	cifar10 \
	fc-tanh \
	mse \
	5e-5 \
	20000 \
	--loss_goal 0.05 \
	--neigs 4 \
	--beta1 0.9 --beta2 0.99 \
	--eig_freq 50 \
	--iterate_freq 50 \
	--save_freq 50 \
	--save_model true

