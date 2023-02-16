# Adaptive_Coalescence_of_Generative_Models
 This repository is the official implementation of ``Continual Learning of  Generative Models with Limited Data: From Wasserstein-1 Barycenter to Adaptive Coalescence''

## Requirements
Please use requirements_conda.txt to create a conda environment and use requirements_pip.txt to create a pip environment.
```conda setup
conda env create -f environment.yml
conda activate [name]
```
or 
```pip setup
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Training
Since many simulation results for different scenarios are presented in the paper, we implemented a lot of easy to use commands. In particular, we use [training_instructions.txt] file to describe certain experiment setups. We have provided a sample document in the [instruction_files] folder. This particular setup, performs 5 fast adaptation, 5 Edge-Only and 5 Transferring GANs experiments in order. Then, generates sample data for computing FID and IS scores. Finally, it computes FID and IS scores. Please modify the [training_instructions.txt] content to try out different experiments.

[Classes.txt] file contains the list of classes and empty rows for [train.py] to save paths of checkpoint files. 

[figure_setup.txt] file contains all the required information for plotting results. First row contains xlabel, ylabel and title. Second line contains the list of legend texts split by `;'. Third row contains the paths to x axis tick values. Forth row contains the paths to y values. All of these values are automatically created when [train.py] is run.

To run experiments, please adjust [training_instructions.txt] and then run
```train
python train.py
```

You can modify all parameters by simply modifying [training_instructions.txt].



