# Adaptive_Coalescence_of_Generative_Models
 This repository is the official implementation of ``Continual Learning of  Generative Models with Limited Data: From Wasserstein-1 Barycenter to Adaptive Coalescence''

## Requirements
Please change the [name] and [prefix] in environment.yml in accordance with your system. Then, run the following code sequence for conda environment.
```setup
conda env create -f environment.yml
conda activate [name]
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

## Training
Since many simulation results for different scenarios are presented in the paper, we completely explain how to reproduce each plot in all experiments. In addition, we provide a code sequence, which will generate data that is ready to be plotted, for every possible scenario considered in the paper. However, we do not provide a particular code sequence to generate a specific experiment. We believe reproducing each experiment with the inclusive code sequence presented here is managable.

We used 4 different projects for MNIST, CIFAR10, CIFAR100 and LSUN. However, same environment works for all. We use three .txt files to describe a particular sequence of training. 

[Classes.txt] file contains the list of classes and empty rows for [train.py] to save paths of checkpoint files. 

[Instructions.txt] file contains all the instructions to be executed in [train.py]. Each row corresponds to a training task. For instance, the command ``Pretrain;Full; 0 1 2 3 4; 10'' trains a full precision generative model using the classes 0, 1, 2, 3, 4 and saves the path to a checkpoint, containing model parameters, at row 10 of [Classes.txt].

[figure_setup.txt] file contains all the required information for plotting results. First row contains xlabel, ylabel and title. Second line contains the list of legend texts split by `;'. Third row contains the paths to x axis tick values, which are created by running [FIDScore.py]. Forth row contains the paths to y values, which are again created by running [FIDScore.py]. 

Before starting to train a model, please run the following code for MNIST, CIFAR10, CIFAR100 and LSUN, which will download datasets and arrange them as required:
```train
python Converter.py
```
For the experiment Figure 15(b), run 
```train
python Converter_SameClass.py
```
To train a single row in [Instructions.txt] for CIFAR10, run the following code:
```train
python train.py --cuda True --batch_size 64 --generator_iters 1 --epoch_num 5001 --FID_BATCH_SIZE 1000 --beta_1 0.5 --beta_2 0.999 --epsilon 0.0000001 --learning_rate 0.0001 --critic_iter 5 --lambda_val 10 --noise_dim_0 100 data_folder CIFAR10_data --workers 2 --subsample 1000 --sample_number 20000 --save_interval 250
```
or simply run 
```train
python train.py
```
as the above values are default.

To train a single row in [Instructions.txt] for MNIST, run the following code:
```train
python train.py --cuda True --batch_size 256 --generator_iters 1 --epoch_num 1001 --FID_BATCH_SIZE 1000 --beta_1 0.5 --beta_2 0.999 --epsilon 0.0000001 --learning_rate 0.0001 --critic_iter 5 --lambda_val 10 --noise_dim_0 100 data_folder MNIST_data --workers 2 --subsample 1000 --sample_number 5000 --save_interval 50
```
or simply run 
```train
python train.py
```
as the above values are default.

The following code sequence along with provided .txt files generates all types of results presented in the paper. Other experiments can be reproduced by just changing [Instructions.txt], [Classes.txt] and input parameters in config.py. Please note that you can stop the running code after each stage as the checkpoint file will be saved and your next run will automatically start training the next task in [Instructions.txt] (Please make sure to delete corresponding folders of incomplete tasks.)

For CIFAR10 training; locate [train.py] under CIFAR10 folder, then run the following code sequence:
```train
python train.py --cuda True --epoch_num 5001 --sample_number 50000
```
This will generate plot data for a particular set of input parameters. You can rerun this code with different input parameters after changing [Instructions.txt] (please use different values in last column of each row) to reproduce the same experiment results. 

To visualize the generated data please run the following code:
```eval
python FIDScore.py --folder_path "CIFAR_Simulations/Folder_BaryTransfer_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]Full14/FID_Images" --sample_number 500
```
You can adjust texts related to figure in [figure_setup.txt]. You should repeat code above for different folder_path values if you want to plot more than one curve in a single figure. Once you run the code above for all folder_paths you desired, please modify [figure_setup.txt] accordingly and then run the code below to visualize them altogether:
```eval
python Plot_Figures.py
```
For MNIST, you can use the same procedure above. Below you can find the full description for MNIST:

-Locate [train.py] under MNIST folder

-Run the code: 
```train
python train.py --cuda True --epoch_num 1001 --sample_number 60000
```
-Run the code multiple times (with different folder_paths) to produce plot data for different experiments:
```eval
python FIDScore.py --folder_path "MNIST_Simulations/Folder_BaryTransfer_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]Full14/FID_Images" --sample_number 500 --select modified_fid_score
```
-Adjust [figure_setup.txt] accordingly.

-Run the code:
```eval
python Plot_Figures.py
```

For LSUN and CIFAR100, please follow the similar steps mentioned above.

For W1GAN-W2GAN comparisons in the supplementary material, please use the folder ``W2GAN-W1GAN''. Similar to previous experiments, Instructions.txt includes the commands for training. The instruction list for generating all the experiments in Fig. 6 are included in Instructions.txt. For training all 4 GANs;
-Run the code: 
```train
python train2.py
```
For evaluation, please follow the following sequence:
-Adjust the input_folder = 'folder path' and 'classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' variables.
-Run the code:
```eval
python Image_Generator.py
```
-Adjust folder_path = 'path to generated images' variable.
-Run the code:
```eval
python FIDScore.py
```
-Run the code: 
```train
python train3.py
```
For evaluation, please follow the following sequence:
-Adjust the input_folder = 'folder path' and 'classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' variables.
-Run the code:
```eval
python Image_Generator.py
```
-Adjust folder_path = 'path to generated images' variable.
-Run the code:
```eval
python FIDScore.py
```
-Adjust [figure_setup.txt] accordingly.

-Run the code:
```eval
python Plot_Figures.py
```

## Evaluation

For generating images from an already trained model, please use the following codes (for MNIST and CIFAR, respectively):
```eval
python evaluate.py --cuda True --data_folder Generated_Images --samples_number 200 --checkpoint_path MNIST_Simulations/Folder_[0, 1, 2, 3, 4]/checkpoint_Iter_1000
```
```eval
python evaluate.py --cuda True --data_folder Generated_Images --samples_number 200 --checkpoint_path CIFAR_Simulations/Folder_[0, 1, 2, 3, 4]/checkpoint_Iter_5000
```
## Pre-trained Models

Alternatively, you can access our pre-trained models at the link below:
```link
https://drive.google.com/drive/folders/1j_q224lTRM847NB_KSuDfcciKPjDD87p?usp=sharing
```









