import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=100001)
    parser.add_argument('--FID_BATCH_SIZE', type=int, default=2000)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-07)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--critic_iter', type=int, default=5)
    parser.add_argument('--lambda_val', type=int, default=10)
    parser.add_argument('--noise_dim_0', type=int, default=100)
    parser.add_argument('--data_folder', type=str, default='CIFAR10_data')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--subsample', type=int, default=0)
    parser.add_argument('--sample_number', type=int, default=50000)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--ensemble_sample_number', type=int, default=10000)

    args = parser.parse_args()

    setattr(args, 'noise_dim', (args.noise_dim_0, 1, 1))

    with open("Classes.txt") as f:
        file_classes = f.read().splitlines()
    train_classes = []
    for line in file_classes:
        try:
            train_classes.append(int(line))
        except ValueError:
            train_classes.append(line)
    f.close()
    setattr(args, 'train_classes', train_classes)

    file = open("Instructions.txt", "r")
    ins = []
    for line in file:
        ins_temp = []
        fields = line.split(";")
        nums = fields[2].split()
        temp = []
        for k in range(len(nums)):
            temp.append(int(nums[k]))
        temp = np.array(temp)
        ins_temp.append(fields[0])
        ins_temp.append(fields[1])
        ins_temp.append(temp)
        ins_temp.append(int(fields[3]))
        try:
            ins_temp.append(int(fields[4]))
        except IndexError:
            pass
        ins.append(ins_temp)
    file.close()
    setattr(args, 'instructions', ins)

    return args


def parse_args_eval():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    parser.add_argument('--data_folder', type=str, default='Generated_Images')
    parser.add_argument('--sample_number', type=int, default=200)
    parser.add_argument('--checkpoint_path', type=str,
                        default='CIFAR_Simulations/Folder_[0, 1, 2, 3, 4]/checkpoint_Iter_9')

    args = parser.parse_args()

    return args


def parse_args_fid():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--folder_path', type=str,
                        default='CIFAR_Simulations/Folder_Transfer_Full19/FID_Images')
    parser.add_argument('--sample_number', type=int, default=500, help='Number of samples at each split')
    parser.add_argument('--select', type=str, default='',
                        help='leave empty for inception_v3 or use "modified_fid_score" for proposed network.')

    args = parser.parse_args()

    file = open("figure_setup.txt", "r")
    labels = []
    legend_text = []
    iteration_files = []
    fid_files = []
    k = 0
    file = file.read().splitlines()
    for line in file:
        fields = line.split(";")
        if k == 0:
            labels.append(fields[0])
            labels.append(fields[1])
            labels.append(fields[2])
        elif k == 1:
            legend_text = fields
        elif k == 2:
            iteration_files = fields
        elif k == 3:
            fid_files = fields
        k += 1
    setattr(args, 'labels', labels)
    setattr(args, 'legend_text', legend_text)
    setattr(args, 'iteration_files', iteration_files)
    setattr(args, 'fid_files', fid_files)

    return args
