#!/bin/bash

# Run each command sequentially in the background and append logs to eval_log.log
CUDA_VISIBLE_DEVICES=3 python3 -m experiments.evaluate --alg_name=ROME-Multi --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-smooth.json >> eval_log-smooth.log 2>&1 &
wait

