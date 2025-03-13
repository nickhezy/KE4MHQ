import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution
import json
import time
import os
import sys

import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimization

set_seed(42)




def top_k_next_tokens(model, tok, prompts, k=10, max_tokens=10):
    # Tokenize input prompt
    inputs = tok(prompts, return_tensors="pt")

    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    # Get model logits
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Get the last token logits
    last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

    # Get the top k token indices and probabilities
    probs = torch.softmax(last_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

    # Convert token indices to actual words
    top_k_tokens = [tok.decode([idx]) for idx in top_k_indices[0].tolist()]

    input_length = inputs["input_ids"].shape[1]  # Length of the input prompt

    # Greedy decoding using model.generate
    greedy_output = model.generate(
        inputs["input_ids"],  # Input token IDs
        max_length=input_length + max_tokens,  # Maximum total length
        num_beams=1,  # Greedy search (no beam search)
        do_sample=False,  # Disable sampling for greedy decoding
        early_stopping=True,  # Stop if EOS token is generated
        pad_token_id=tok.eos_token_id,  # Use EOS token for padding
    )
    
    generated_tokens = greedy_output[0, input_length:] 

    # Decode the generated tokens to text
    greedy_text = tok.decode(generated_tokens, skip_special_tokens=True)

    # Return a dictionary with the top k tokens, their probabilities, and the greedy generation
    return {
        "top_k_tokens": {top_k_tokens[i]: top_k_probs[0, i].item() for i in range(k)},
        "greedy": greedy_text,
    }


def eval_editing(model, case, rel_prompts, tok, save_dir=None):
    model_name = model.name_or_path

    print("\n\n"+5*"++++++++++++++++++++++")
    print(f"Evaluating case_id: {case['case_id']}")


    # generate top-k dictionary to the questions and save the dictionary to a json file under folder: top_k_dict
    # add all single-hop questions 
    questions = [new_single_hop["question"] for new_single_hop in case["new_single_hops"]]
    # add MHQ question
    questions.append(case["questions"][0])

    rel_context = [rel_prompts[trip[1]] for trip in case["orig"]["new_triples"]]
    rel_context.append(rel_prompts[case["orig"]["new_triples"][1][1]])

    answers = [new_single_hop["answer"] for new_single_hop in case["new_single_hops"]]
    answers.append(case["new_answer"])
    
    # one dict for each question. each dict question has keys: question, answer, results
    results = [
        {
            "case_id": case["case_id"],
            # "hop": case["hop"],
            "requested_rewrites": case["requested_rewrite"],
        }
    ]


    for i in range(len(questions)):
        results.append(
            {
                "context": rel_context[i],
                "question": questions[i],
                "model_responses": top_k_next_tokens(
                    model, 
                    tok, 
                    rel_context[i] + "\nQ: " + questions[i] + " A:", 
                    k=10
                ),
                "answer": answers[i],
                

            }
        )
    results[-1]["new_answer_alias"] = case["new_answer_alias"]
    # check correctness of the MHQ answer
    results[-1]["correct"] = results[-1]["model_responses"]["greedy"].lstrip().\
                            startswith((case["new_answer"],)+tuple(case["new_answer_alias"]))

        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"multi-edit-results/{save_dir}/{model_name[-4:]}_id_{case['case_id']}.json", "w") as f:
            json.dump(results, f)
        print("Evaluation result saved to: ", f"{save_dir}/{model_name[-4:]}_id_{case['case_id']}.json")

    return results[-1]["correct"]