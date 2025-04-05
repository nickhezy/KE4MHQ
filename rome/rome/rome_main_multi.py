from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None




def apply_multi_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    return_orig_weights_device="cuda",
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes while managing GPU memory efficiently.
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}
    updates = {}  # Store updates in CPU memory

    orig_layers = hparams.layers  # Comment out when eval
    for i, request in enumerate(requests):
        hparams.layers = orig_layers[i]  # Comment out when eval

        for layer in sorted(hparams.layers):
            deltas = execute_multi_rome(model, tok, request, hparams, layer)

            with torch.no_grad():
                for w_name, (delta_u, delta_v) in deltas.items():
                    upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                    # Move update matrix to CPU to free GPU memory
                    upd_matrix = upd_matrix.to("cpu")
                    # Store update on CPU to apply later
                    updates[w_name] = upd_matrix

                    if return_orig_weights and w_name not in weights_copy:
                        weights_copy[w_name] = w.detach().clone().to(return_orig_weights_device)

            # **Free GPU Memory after each layer update**
            torch.cuda.empty_cache()

    # **Step 2: Apply all updates (moving back to GPU only when needed)**
    with torch.no_grad():
        for w_name in list(updates.keys()):  #  Iterate over a static copy of keys to avoid dict size change
            upd_matrix = updates[w_name]  # Fetch the stored update matrix

            w = nethook.get_parameter(model, w_name)

            # Move update matrix back to GPU
            upd_matrix = upd_matrix.to(w.device)

            # Ensure update matrix matches weight shape
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            # Apply update
            w[...] += upd_matrix

            # Remove the entry safely
            del updates[w_name]
            torch.cuda.empty_cache()

    return model, weights_copy


def execute_multi_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    # Compute rank-1 update matrix
    left_vector: torch.Tensor = compute_u(
        model,
        tok,
        request,
        hparams,
        layer,
        get_context_templates(model, tok, hparams.context_template_length_params),
    )
    print("Left vector shape:", left_vector.shape)
    # Get model's device
    device = next(model.parameters()).device

    # Compute right_vector and ensure it is on the correct device
    right_vector: torch.Tensor = compute_v(
        model,
        tok,
        request,
        hparams,
        layer,
        left_vector.to(device), 
        get_context_templates(model, tok, hparams.context_template_length_params),
    ).to(device)  # Ensure final tensor is on the correct device
    print("Right vector shape:", right_vector.shape)

    with torch.no_grad():
        # Determine correct transposition of delta matrix
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        # Update model weights and record desired changes in `delta` variable
        weights[weight_name][...] += upd_matrix
        deltas[weight_name] = (
            left_vector.detach(),
            right_vector.detach(),
        )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {layer}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE