import json
import os
import sys


# folder_dir = "KE4MHQ/rome/Hop1-Eval-5-10-15-20"
# get all the subdirectories in the folder
subdirs = [x[0] for x in os.walk(folder_dir)]
print(subdirs)


def write_answer(dict_list):
    resp = next(iter(dict_list[-1]["top_k_responses"])).strip()
    if dict_list[-1]["answer"].startswith(resp):
        dict_list[-1]["correctness"] = 1
        return 1
    else:
        dict_list[-1]["correctness"] = 0
        return 0


print("correctness: total_correctness(hop2_edit_correctness)/number_of_data for editing_which_layers")
for subdir in subdirs:
    # get a list of json files in the folder
    json_files = [pos_json for pos_json in os.listdir(subdir) if pos_json.endswith('.json')]
    total = len(json_files)
    if not total:
        continue
    c = 0
    hop1_c = 0
    for jf in json_files:
        with open(os.path.join(subdir, jf)) as f:
            data = json.load(f)
        correct = write_answer(data)
        if correct:
            c += 1
            if data[0]["hop"][0] == 1:
                hop1_c += 1
        with open(os.path.join(subdir, jf), 'w') as f:
            json.dump(data, f)
        # print(f"Done writing {jf}")
    print(f"correctness: {c}({hop1_c})/{total} for {subdir}")



