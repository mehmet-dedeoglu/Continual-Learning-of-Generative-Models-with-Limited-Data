import argparse
from supplementary.options import Options


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--instruction_input', type=str,
                        default='instruction_files/training_instructions.txt')

    parser.add_argument('--classes_input', type=str, default='instruction_files/classes.txt')
    parser.add_argument('--mode', type=str, default='W1GAN_train', help='Availability of cuda')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    parser.add_argument('--dataset_prefix', type=str, default='../Datasets/', help='Dataset locations w.r.t. code loc.')
    parser.add_argument('--cuda_index', type=int, default=0, help='Cuda device index')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=2001)
    parser.add_argument('--FID_BATCH_SIZE', type=int, default=1000)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-07)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--critic_iter', type=int, default=5)
    parser.add_argument('--lambda_val', type=int, default=10)
    parser.add_argument('--noise_dim_0', type=int, default=100)
    parser.add_argument('--checkpoint_save', type=str, default='All')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--subsample', type=int, default=8)  # 100
    parser.add_argument('--sample_number', type=int, default=5000)  # 5000
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--channel_number', type=int, default=1)
    parser.add_argument('--select', type=str, default='W1GAN')  # W2GAN, W1GAN
    parser.add_argument('--data_folder', type=str, default=' ')  #MNIST_data
    parser.add_argument('--score_batch_size', type=int, default=256)
    parser.add_argument('--score_runs', type=int, default=2)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    args = parser.parse_args()
    config = Options().parse()

    # parser.add_argument('--ensemble_sample_number', type=int, default=6000)
    args.config = config

    setattr(args, 'noise_dim', (args.noise_dim_0, 1, 1))

    with open(args.classes_input) as f:
        file_classes = f.read().splitlines()
    train_classes = []
    for line in file_classes:
        try:
            train_classes.append(int(line))
        except ValueError:
            train_classes.append(line)
    f.close()
    setattr(args, 'train_classes', train_classes)

    file = open(args.instruction_input, "r")
    ins = []
    for line in file:
        temp = []
        line0 = line.split("/")[0]
        fields = line0.split(":")
        for line1 in fields:
            temp1 = []
            fields1 = line1.split(";")
            for line2 in fields1:
                temp2 = []
                fields2 = line2.split(",")
                for line3 in fields2:
                    fields3 = line3.split("-")
                    try:
                        temp2.append(int(fields3[0]))
                    except ValueError:
                        temp2.append(fields3[0])
                    try:
                        try:
                            fields3[1] = int(fields3[1])
                        except ValueError:
                            pass
                        temp2.append(fields3[1])
                        try:
                            temp2.append(int(fields3[2]))
                        except IndexError:
                            pass
                    except IndexError:
                        pass
                temp1.append(temp2)
            temp.append(temp1)
        ins.append(temp)
    file.close()
    setattr(args, 'instructions', ins)

    # args = get_score_params(args)

    return args


def get_score_params(args_):

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
    setattr(args_, 'labels', labels)
    setattr(args_, 'legend_text', legend_text)
    setattr(args_, 'iteration_files', iteration_files)
    setattr(args_, 'fid_files', fid_files)
    setattr(args_, 'score_folder_path', '')  #MNIST_Simulations/Folder_EdgeOnly_[8, 9]Full20/FID_Images
    setattr(args_, 'score_sample_number', 500)
    setattr(args_, 'score_select', 'modified_fid_score')

    return args_


def parse_args_eval():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')
    parser.add_argument('--data_folder', type=str, default='Generated_Images')
    parser.add_argument('--sample_number', type=int, default=200)
    parser.add_argument('--select', type=str, default='W2GAN')
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default='MNIST_Simulations/Stage_17/checkpoints/checkpoint_Iter_100')
    parser.add_argument('--checkpoint_path', type=str,
                        default='W2Sims/stage18/checkpoint_Iter_10000')
    parser.add_argument('--template_path', type=str,
                        default='MNIST_data_all/Label_0_.png')
    parser.add_argument('--channel_number', type=int, default=1)
    config = Options().parse()

    args = parser.parse_args()
    args.config = config

    return args


def parse_args_fid():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--folder_path', type=str,
                        default='figures/100_samples_overlapping')
    parser.add_argument('--sample_number', type=int, default=500, help='Number of samples at each split')
    parser.add_argument('--smooth_samples', type=int, default=2, help='Number of samples at each split')
    # parser.add_argument('--select', type=str, default='modified_fid_score',
    #                     help='leave empty for inception_v3 or use "modified_fid_score" for proposed network.')

    args = parser.parse_args()
    file = open("../instruction_files/figure_setup.txt", "r")
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
