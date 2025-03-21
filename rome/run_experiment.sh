#!/bin/bash

# Run each command sequentially in the background and append logs to eval_log.log
CUDA_VISIBLE_DEVICES=3 python3 -m experiments.evaluate --alg_name=ROME-Multi --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-5-10-20.json >> eval_log-5-10-20.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=3 python3 -m experiments.evaluate --alg_name=ROME-Multi --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-5-15.json >> eval_log-5-15.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=3 python3 -m experiments.evaluate --alg_name=ROME-Multi --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-5-20.json >> eval_log-5-20.log 2>&1 &
wait

