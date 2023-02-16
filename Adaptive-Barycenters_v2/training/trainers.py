from utils.data_loader import move_data, move_data2, get_data_loader
import torch.optim as optim
import torch
import supplementary.networks as networks
import itertools
from archs.wb_networks import CNN_Dis_gp, CNN_Gen_gp, CNN_Gen_Ter_Thres_gp, CNN_Dis_Ter_Thres_gp
from training.WGAN_Bary import WGAN_GP
from training.W2GAN_Bary import W2GAN_GP
from training.WGAN_Bary_Thres import WGAN_GP_Thres
import os
import shutil
import warnings
warnings.filterwarnings("ignore")


def W1GAN_trainer(args, instructions, k):
    # This is a training stage. Create stage folder
    sim_folder = args.dataset_prefix + args.dataset + '_Simulations'
    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
        setup_file = sim_folder + '/setup.txt'
        f = open(setup_file, "w+")
        f.write(str(args))
        f.close()
    stage_folder = sim_folder + '/Stage_' + str(k)
    # Check if this stage is completed earlier.
    final_file_name = stage_folder + '/checkpoints/checkpoint_Iter_' + str(args.epoch_num - 1)
    if os.path.isfile(final_file_name):
        # This stage is completed earlier. Skip this stage.
        print('This stage is completed earlier. Skip this stage.')
    else:
        # This stage is not completed earlier.
        if not os.path.exists(stage_folder):
            os.mkdir(stage_folder)
        else:
            shutil.rmtree(stage_folder)
            os.mkdir(stage_folder)
        # Create dataset folders for each discriminator
        numOfDiscriminators = len(instructions[k][0])
        folder_name = []
        datasets = []
        sampleNumberSave = 10000000
        dataset_locations = []
        for disc_index in range(numOfDiscriminators):
            numOfClasses = int(len(instructions[k][0][disc_index])/2)
            folder_name.append(stage_folder + '/Data_folder_' + str(disc_index))
            totalSampleNumber = 0
            for class_index in range(numOfClasses):
                dataset_index = instructions[k][0][disc_index][2*class_index]
                dataset_sample = instructions[k][0][disc_index][2*class_index+1]
                print(folder_name[disc_index])
                dataset_loc = move_data(args, dataset_index, dataset_sample, folder_name[disc_index])
                # Compute total sample number to adjust batch size
                if instructions[k][0][disc_index][2*class_index+1] == 'All':
                    totalSampleNumber += 100000  # 100000 is an arbitrarily large number
                else:
                    totalSampleNumber += instructions[k][0][disc_index][2*class_index+1]
            if totalSampleNumber < sampleNumberSave:
                sampleNumberSave = totalSampleNumber
            dataset_locations.append(dataset_loc)

        # Check if dataset size is smaller than batch size.
        old_batch_size = args.batch_size
        if sampleNumberSave < old_batch_size:
            args.batch_size = sampleNumberSave
        # Load datasets for discriminators
        for disc_index in range(numOfDiscriminators):
            # Generate data loader for this discriminator
            datasets.append(get_data_loader(args, dataset_locations[disc_index]))

        # Create initial generator and discriminator models
        g_mod = CNN_Gen_gp(args.channel_number)  # Full model (previous model parameters will be transferred here.)
        d_mod = []
        for disc_index in range(numOfDiscriminators):
            d_mod.append(CNN_Dis_gp(args.channel_number))
        # Load initial models if available
        try:
            model_exists = instructions[k][1][0][1]  # Initial model exists
            # Load models from discriminator descriptions
            for disc_index in range(numOfDiscriminators):
                try:
                    # We have an initial disc. model only if the length is odd.
                    if len(instructions[k][0][disc_index]) % 2 == 1:
                        model_loc = args.train_classes[instructions[k][0][disc_index][-1]]
                        checkpoint = torch.load(model_loc)
                        # ['modelD'][0] is either pretrained disc. model or the barycenter disc. model
                        d_mod[disc_index].load_state_dict(checkpoint['modelD'][0])
                except IndexError or UnboundLocalError:
                    pass
            if model_exists != 'True':  # There is an initial generator model
                # Check for generator models
                try:
                    model_loc = args.train_classes[instructions[k][1][0][1]]
                    checkpoint = torch.load(model_loc)
                    g_mod.load_state_dict(checkpoint['modelG'])
                except IndexError or UnboundLocalError:
                    print('An unexpected error occurred: Could not find checkpoint file.')
        except IndexError:  # No initial models
            pass
        # Create optimizers for discriminators and the generator
        g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
        d_opt = []
        for disc_index in range(numOfDiscriminators):
            d_opt.append(optim.Adam(d_mod[disc_index].parameters(), lr=args.learning_rate,
                                    betas=(args.beta_1, args.beta_2)))

        GAN = WGAN_GP(args, stage_folder, g_mod, d_mod, g_opt, d_opt)
        loc_string = GAN.train(datasets)
        for locs_ in dataset_locations:
            shutil.rmtree(locs_)

        args.batch_size = old_batch_size
        args.train_classes[instructions[k][1][0][0]] = loc_string
        with open(args.classes_input, 'w') as f:
            for item in args.train_classes:
                f.write("%s\n" % item)


def W2GAN_trainer(args, instructions, k):
    # This is a training stage. Create stage folder
    sim_folder = args.dataset_prefix + args.dataset + '_Simulations'
    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    stage_folder = sim_folder + '/Stage_' + str(k)
    # Check if this stage is completed earlier.
    final_file_name = stage_folder + '/checkpoints/checkpoint_Iter_' + str(args.epoch_num - 1)
    if os.path.isfile(final_file_name):
        # This stage is completed earlier. Skip this stage.
        print('This stage is completed earlier. Skip this stage.')
    else:
        # This stage is not completed earlier.
        if not os.path.exists(stage_folder):
            os.mkdir(stage_folder)
        else:
            shutil.rmtree(stage_folder)
            os.mkdir(stage_folder)
        # Create dataset folders for each discriminator
        numOfDiscriminators = len(instructions[k][0])
        folder_name = []
        datasets = []
        sampleNumberSave = 10000000
        dataset_locations = []
        for disc_index in range(numOfDiscriminators):
            numOfClasses = int(len(instructions[k][0][disc_index]) / 2)
            folder_name.append(stage_folder + '/Data_folder_' + str(disc_index))
            totalSampleNumber = 0
            for class_index in range(numOfClasses):
                dataset_index = instructions[k][0][disc_index][2 * class_index]
                dataset_sample = instructions[k][0][disc_index][2 * class_index + 1]
                dataset_loc = move_data2(args, dataset_index, dataset_sample, folder_name[disc_index])
                # Compute total sample number to adjust batch size
                if instructions[k][0][disc_index][2 * class_index + 1] == 'All':
                    totalSampleNumber += 100000  # 100000 is an arbitrarily large number
                else:
                    totalSampleNumber += instructions[k][0][disc_index][2 * class_index + 1]
            if totalSampleNumber < sampleNumberSave:
                sampleNumberSave = totalSampleNumber
            dataset_locations.append(dataset_loc)

        # Check if dataset size is smaller than batch size.
        old_batch_size = args.batch_size
        if sampleNumberSave < old_batch_size:
            args.batch_size = sampleNumberSave
        # Load datasets for discriminators
        for disc_index in range(numOfDiscriminators):
            # Generate data loader for this discriminator
            datasets.append(get_data_loader(args, dataset_locations[disc_index]))

        # Create initial generator and discriminator models
        g_mod = networks.get_g(args.config)
        # Full model (previous model parameters will be transferred here.)
        df_mod = []
        db_mod = []
        for disc_index in range(numOfDiscriminators):
            phi, eps = networks.get_d(args.config), networks.get_d(args.config)
            df_mod.append(phi)
            db_mod.append(eps)
        # Load initial models if available
        try:
            model_exists = instructions[k][1][0][1]  # Initial model exists
            # Load models from discriminator descriptions
            for disc_index in range(numOfDiscriminators):
                try:
                    # We have an initial disc. model oly if the length is odd.
                    if len(instructions[k][0][disc_index]) % 2 == 1:
                        model_loc = args.train_classes[instructions[k][0][disc_index][-1]]
                        checkpoint = torch.load(model_loc)
                        # ['modelD'][0] is either pretrained disc. model or the barycenter disc. model
                        df_mod[disc_index].load_state_dict(checkpoint['modelDF'][0])
                        db_mod[disc_index].load_state_dict(checkpoint['modelDB'][0])
                except IndexError or UnboundLocalError:
                    pass
            if model_exists != 'True':  # There is an initial generator model
                # Check for generator models
                try:
                    model_loc = args.train_classes[instructions[k][1][0][1]]
                    checkpoint = torch.load(model_loc)
                    g_mod.load_state_dict(checkpoint['modelG'])
                except IndexError or UnboundLocalError:
                    print('An unexpected error occurred: Could not find checkpoint file.')
        except IndexError:  # No initial models
            pass
        # Create optimizers for discriminators and the generator
        g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
        d_opt = []
        for disc_index in range(numOfDiscriminators):
            d_opt.append(networks.get_optim(itertools.chain(list(df_mod[disc_index].parameters()),
                                                            list(db_mod[disc_index].parameters())),
                                            args.config.d_lr, args.config))

        GAN = W2GAN_GP(args, stage_folder, g_mod, df_mod, db_mod, g_opt, d_opt)
        loc_string = GAN.train(datasets)

        args.batch_size = old_batch_size
        args.train_classes[instructions[k][1][0][0]] = loc_string
        with open(args.classes_input, 'w') as f:
            for item in args.train_classes:
                f.write("%s\n" % item)


def W1GAN_trainer_with_W2Nets(args, instructions, k):
    # This is a training stage. Create stage folder
    sim_folder = args.dataset_prefix + args.dataset + '_Simulations'
    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    stage_folder = sim_folder + '/Stage_' + str(k)
    # Check if this stage is completed earlier.
    final_file_name = stage_folder + '/checkpoints/checkpoint_Iter_' + str(args.epoch_num - 1)
    if os.path.isfile(final_file_name):
        # This stage is completed earlier. Skip this stage.
        print('This stage is completed earlier. Skip this stage.')
    else:
        # This stage is not completed earlier.
        if not os.path.exists(stage_folder):
            os.mkdir(stage_folder)
        else:
            shutil.rmtree(stage_folder)
            os.mkdir(stage_folder)
        # Create dataset folders for each discriminator
        numOfDiscriminators = len(instructions[k][0])
        folder_name = []
        datasets = []
        sampleNumberSave = 10000000
        dataset_locations = []
        for disc_index in range(numOfDiscriminators):
            numOfClasses = int(len(instructions[k][0][disc_index]) / 2)
            folder_name.append(stage_folder + '/Data_folder_' + str(disc_index))
            totalSampleNumber = 0
            for class_index in range(numOfClasses):
                dataset_index = instructions[k][0][disc_index][2 * class_index]
                dataset_sample = instructions[k][0][disc_index][2 * class_index + 1]
                dataset_loc = move_data2(args, dataset_index, dataset_sample, folder_name[disc_index])
                # Compute total sample number to adjust batch size
                if instructions[k][0][disc_index][2 * class_index + 1] == 'All':
                    totalSampleNumber += 100000  # 100000 is an arbitrarily large number
                else:
                    totalSampleNumber += instructions[k][0][disc_index][2 * class_index + 1]
            if totalSampleNumber < sampleNumberSave:
                sampleNumberSave = totalSampleNumber
            dataset_locations.append(dataset_loc)

        # Check if dataset size is smaller than batch size.
        old_batch_size = args.batch_size
        if sampleNumberSave < old_batch_size:
            args.batch_size = sampleNumberSave
        # Load datasets for discriminators
        for disc_index in range(numOfDiscriminators):
            # Generate data loader for this discriminator
            datasets.append(get_data_loader(args, dataset_locations[disc_index]))

        # Create initial generator and discriminator models
        g_mod = networks.get_g(args.config)
        # Full model (previous model parameters will be transferred here.)
        d_mod = []
        for disc_index in range(numOfDiscriminators):
            phi = networks.get_d(args.config)
            d_mod.append(phi)
        # Load initial models if available
        try:
            model_exists = instructions[k][1][0][1]  # Initial model exists
            # Load models from discriminator descriptions
            for disc_index in range(numOfDiscriminators):
                try:
                    # We have an initial disc. model oly if the length is odd.
                    if len(instructions[k][0][disc_index]) % 2 == 1:
                        model_loc = args.train_classes[instructions[k][0][disc_index][-1]]
                        checkpoint = torch.load(model_loc)
                        # ['modelD'][0] is either pretrained disc. model or the barycenter disc. model
                        d_mod[disc_index].load_state_dict(checkpoint['modelD'][0])
                except IndexError or UnboundLocalError:
                    pass
            if model_exists != 'True':  # There is an initial generator model
                # Check for generator models
                try:
                    model_loc = args.train_classes[instructions[k][1][0][1]]
                    checkpoint = torch.load(model_loc)
                    g_mod.load_state_dict(checkpoint['modelG'])
                except IndexError or UnboundLocalError:
                    print('An unexpected error occurred: Could not find checkpoint file.')
        except IndexError:  # No initial models
            pass
        # Create optimizers for discriminators and the generator
        g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
        d_opt = []
        for disc_index in range(numOfDiscriminators):
            d_opt.append(networks.get_optim(itertools.chain(list(d_mod[disc_index].parameters())),
                                            args.config.d_lr, args.config))

        GAN = WGAN_GP(args, stage_folder, g_mod, d_mod, g_opt, d_opt)
        loc_string = GAN.train(datasets)

        args.batch_size = old_batch_size
        args.train_classes[instructions[k][1][0][0]] = loc_string
        with open(args.classes_input, 'w') as f:
            for item in args.train_classes:
                f.write("%s\n" % item)
