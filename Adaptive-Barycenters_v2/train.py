from utils.config import parse_args
from utils.data_loader import generate_images
import torch
from training.trainers import W1GAN_trainer, W2GAN_trainer, W1GAN_trainer_with_W2Nets
from score_compute import score_manager
import warnings
warnings.filterwarnings("ignore")


def main(args):
    k = 0
    instructions = args.instructions
    while k < len(instructions):
        if instructions[k][0][0][0] == 'cuda':
            args.cuda = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'batch_size':
            args.batch_size = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'generator_iters':
            args.generator_iters = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'critic_iter':
            args.critic_iter = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'epoch_num':
            args.epoch_num = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'beta_1':
            args.beta_1 = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'beta_2':
            args.beta_2 = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'epsilon':
            args.epsilon = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'learning_rate':
            args.learning_rate = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'lambda_val':
            args.lambda_val = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'dataset':
            args.dataset = instructions[k][1][0][0]
            if args.dataset == 'MNIST':
                args.channel_number = 1
                # args.data_folder = 'MNIST_data'
            elif args.dataset == 'CIFAR10':
                args.channel_number = 3
                # args.data_folder = 'CIFAR10_data'
            elif args.dataset == 'CIFAR100':
                args.channel_number = 3
                # args.data_folder = 'CIFAR100_data'
            elif args.dataset == 'LSUN':
                args.channel_number = 3
                # args.data_folder = 'LSUN_data'
            elif args.dataset == 'CIFAR10+CIFAR100':
                args.channel_number = 3
        elif instructions[k][0][0][0] == 'subsample':
            args.subsample = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'save_interval':
            args.save_interval = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'FID_BATCH_SIZE':
            args.FID_BATCH_SIZE = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'critic_weights':
            args.critic_weights = instructions[k][1][0]
        elif instructions[k][0][0][0] == 'precision':
            args.precision = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'checkpoint_save':
            args.checkpoint_save = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'mode':
            args.mode = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'dnn_type':
            args.dnn_type = int(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'score_split_size':
            args.score_split_size = int(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'score_type':
            args.score_type = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'checkpoint_freq':
            args.checkpoint_freq = instructions[k][1][0][0]
        else:
            if args.mode == 'W1GAN_train':
                W1GAN_trainer(args, instructions, k)
            elif args.mode == 'W2GAN_train':
                W2GAN_trainer(args, instructions, k)
            elif args.mode == 'W1GAN_train_with_W2_DNN':
                W1GAN_trainer_with_W2Nets(args, instructions, k)
            elif args.mode == 'generate':
                _ = generate_images(args, instructions[k])
            elif args.mode == 'score':
                print('Entering score manager...')
                score_manager(args, instructions[k])
            else:
                print('Not implemented.')

        print(str(k) + 'th stage is completed.')
        k = k + 1


if __name__ == '__main__':
    arguments = parse_args()
    print('Device in use is: ' + torch.cuda.get_device_name(arguments.cuda_index))
    main(arguments)
    print('Finished')
