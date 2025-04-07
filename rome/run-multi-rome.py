import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution
import json
import time

from util.eval_greedy import eval_editing,eval_old_ans


import numpy as np
import random
import tqdm

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimization

set_seed(42)

IS_COLAB = False
ALL_DEPS = False
# try:
#     import google.colab, torch, os

#     IS_COLAB = True
#     os.chdir("/content/rome")
#     if not torch.cuda.is_available():
#         raise Exception("Change runtime type to include a GPU.")
# except ModuleNotFoundError as _:
#     pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = 'cpu'
# device = torch.device('cuda:1')
# print(f"Using device: {device}")

ALG_NAME = "ROME-Multi" # alternatively: "ROME"
MODEL_NAME = "EleutherAI/gpt-j-6B" # alternatively: "gpt2-xl"

json_path_dict = {
    "gpt2-xl":"hparams/ROME/gpt2-xl.json",
    "EleutherAI/gpt-j-6B":"hparams/ROME/EleutherAI_gpt-j-6B.json"
}


model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
        device
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token





json_path_dict = {
    "gpt2-xl":"hparams/ROME/gpt2-xl.json",
    "EleutherAI/gpt-j-6B":"hparams/ROME/EleutherAI_gpt-j-6B.json"
}



# context used when testing the editted model
context_file = "dsets/rel-prompts.json"
with open(context_file, "r") as f:
    rel_prompts = json.load(f)



def test_multi_rome(
    layers_to_edit, 
    edit_hop, 
    continue_from=0, 
    max_cases=1e9, 
    save_dir=None, 
    kl_factor=0.0625, 
    v_num_grad_steps=20,
    tgt_first_tok_prob=0.2,
    no_edit=False,
):

    with open(json_path_dict[MODEL_NAME], "r") as f:
        data = json.load(f)
    data["layers"] = layers_to_edit
    data["kl_factor"] = kl_factor
    data["v_num_grad_steps"] = v_num_grad_steps
    data["tgt_first_tok_prob"] = tgt_first_tok_prob
    
    with open(json_path_dict[MODEL_NAME], "w") as f:
        json.dump(data, f, indent=2)

    ds_file = "dsets/ds_classification/"+edit_hop+"_edits.json"
    with open(ds_file, "r") as f:
        mhq_ds = json.load(f)

    print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("layers_to_edit: ", layers_to_edit)
    print("ds_file: ", ds_file)
    correct = 0
    count = 0
    for i in range(len(mhq_ds)):
    # for i in range(100):
        if i < continue_from:
            continue
        if i >= continue_from + max_cases:
            break
        case = mhq_ds[i]
        request = case["requested_rewrite"]
        generation_prompts = []

        print("\n\n"+4*"***********************************************")
        print(f"Request {i+1}, case_id: {case['case_id']}")

        if not no_edit:
            # Restore fresh copy of model
            try:
                with torch.no_grad():
                    for k, v in orig_weights.items():
                        nethook.get_parameter(model, k)[...] = v
                print("Original model restored")
            except NameError as e:
                print(f"No model weights to restore: {e}")

            # Execute rewrite
            model_new, orig_weights = demo_model_editing(
                model, tok, request, generation_prompts, alg_name=ALG_NAME,generate_prompts=False 
                )
            
            eval_fn = eval_editing
        else:
            eval_fn = eval_old_ans

        if save_dir is None:
            # use default save_dir
            save_dir = edit_hop+"-Eval-" + "-".join(map(str, layers_to_edit))

        if eval_fn(model, case, rel_prompts, tok, save_dir=save_dir):
            correct += 1

        count+=1

        print(f"Correct: {correct}/{count}")

    print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))




layers_sweep = [
    # [[8]],
    # [[13]],
    [[18]],
    [[13]],
    [[5,9,13,17,20]],

    # [[5, 20]]

    # [[5,10,20]]
    # [[10,15,20]],
    

]
hop_sweep = [
    "hop2",
    "hop1", 
    # "both"
]

for layers in layers_sweep:
    for hop in hop_sweep:
        test_multi_rome(layers_to_edit=layers, edit_hop=hop)
