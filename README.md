
## Installation Guide
Follow these steps to set up your Jupyter Notebook environment with Conda:

```sh
# Step 1: Create a Conda virtual environment with Python 3.11
conda create -n KE4MHQ_env python=3.9.7

# Step 2: Activate the virtual environment
conda activate KE4MHQ_env

# Step 3: Install dependencies from requirements.txt
pip install -r requirements.txt

# Step 4: Install Jupyter Notebook
pip install jupyter

# Step 5: Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=KE4MHQ_kernel --display-name "Python (KE4MHQ)"
```

## **Final Step: Launch Jupyter in VSCode**
1. Open **VSCode**.
2. Open a **Jupyter Notebook** (`.ipynb` file).
3. Select the kernel **"Python (KE4MHQ)"** from the top-right kernel selector.
4. You’re ready to code! 

## Dataset
A subset from MQuake with classification:

[KE4MHQ/rome/dsets/ds_classification](https://github.com/nickhezy/KE4MHQ/blob/master/rome/dsets/ds_classification/README.txt)

## Run Multi-ROME
[KE4MHQ/Multi-ROME.ipynb](https://github.com/nickhezy/KE4MHQ/blob/master/Multi-ROME.ipynb)

implementation of the algorithm is in [KE4MHQ/rome/rome/rome_main_multi.py](https://github.com/nickhezy/KE4MHQ/blob/master/rome/rome/rome_main_multi.py)

