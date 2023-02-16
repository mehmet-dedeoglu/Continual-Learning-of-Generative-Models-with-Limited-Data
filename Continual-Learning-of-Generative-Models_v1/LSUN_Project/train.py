from config import parse_args
from data_loader import get_data_loader_custom, ensemble
import torch.optim as optim
import torch
from DNN_CIFAR10 import CNN_Dis_gp, CNN_Gen_gp
from DNN_CIFAR10_Thres import CNN_Gen_Ter_Thres_gp, CNN_Dis_Ter_Thres_gp
from WGAN_Bary import WGAN_GP
from WGAN_Bary_Thres import WGAN_GP_Thres
import os
import warnings
warnings.filterwarnings("ignore")


def main(args):
    k = 0
    instructions = args.instructions
    while k < len(instructions):
        if instructions[k][0] == 'Pretrain':
            # Indices must refer to datasets!
            indices = instructions[k][2]
            tr_cls = []
            for it in range(len(indices)):
                tr_cls.append(args.train_classes[indices[it]])
            save_folder = 'LSUN_Simulations/Folder_' + str(tr_cls)
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                # Perform Training
                g_mod = CNN_Gen_gp(3)
                d_mod = [CNN_Dis_gp(3)]
                # Create optimizers for each WGAN model
                g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate,
                                    betas=(args.beta_1, args.beta_2))]
                # Create training object

                gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')
                dataset, _ = get_data_loader_custom(args, args.batch_size, tr_cls, subsample=0)
                dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, tr_cls, subsample=0)
                # Train each pre-trained generator and discriminator models
                loc_string = gan.train(dataset, dataset_fid, discriminator_save_iter=-1)
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'EdgeOnly':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample

            # Indices must refer to datasets!
            indices = instructions[k][2]
            tr_cls = []
            for it in range(len(indices)):
                tr_cls.append(args.train_classes[indices[it]])
            save_folder = 'LSUN_Simulations/Folder_EdgeOnly_' + str(tr_cls) + instructions[k][1] \
                          + str(instructions[k][3])
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                if instructions[k][1] == 'Ternarized_with_Threshold':
                    g_mod = CNN_Gen_Ter_Thres_gp(3)  # Ternarized model
                    all_gparam_new = [param for name, param in g_mod.named_parameters() if not 'delta_th' in name]
                    g_opt_all = optim.Adam(all_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                    th_gparam_new = [param for name, param in g_mod.named_parameters() if 'delta_th' in name]
                    g_opt_th = optim.Adam(th_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    d_mod = [CNN_Dis_Ter_Thres_gp(3)]
                    all_dparam_new = [param for name, param in d_mod[0].named_parameters() if not 'delta_th' in name]
                    th_dparam_new = [param for name, param in d_mod[0].named_parameters() if 'delta_th' in name]
                    d_opt_all = [optim.Adam(all_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                    d_opt_th = [optim.Adam(th_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]

                    gan = WGAN_GP_Thres(args, save_folder, g_mod, d_mod, [g_opt_all, g_opt_th],
                                        [d_opt_all, d_opt_th], fid_state='FID')
                else:
                    g_mod = CNN_Gen_gp(3)  # Full model
                    d_mod = [CNN_Dis_gp(3)]

                if instructions[k][1] != 'Ternarized_with_Threshold':
                    # Create optimizers for each WGAN model
                    g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                    d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate,
                                        betas=(args.beta_1, args.beta_2))]
                    # Create training object
                    gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')

                dataset, _ = get_data_loader_custom(args, args.batch_size, tr_cls, subsample=args.subsample)
                dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, tr_cls, subsample=0)
                # Train each pre-trained generator and discriminator models
                loc_string = gan.train(dataset, dataset_fid, discriminator_save_iter=-1)
                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'FullData':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample

            # Indices must refer to datasets!
            indices = instructions[k][2]
            tr_cls = []
            for it in range(len(indices)):
                tr_cls.append(args.train_classes[indices[it]])
            if len(tr_cls) > 25:
                save_folder = 'LSUN_Simulations/Folder_FullData_Too_Long' + str(tr_cls[:5]) + instructions[k][1]
            else:
                save_folder = 'LSUN_Simulations/Folder_FullData_' + str(tr_cls) + instructions[k][1]
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                if instructions[k][1] == 'Ternarized_with_Threshold':
                    g_mod = CNN_Gen_Ter_Thres_gp(3)  # Ternarized model
                    all_gparam_new = [param for name, param in g_mod.named_parameters() if not 'delta_th' in name]
                    g_opt_all = optim.Adam(all_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                    th_gparam_new = [param for name, param in g_mod.named_parameters() if 'delta_th' in name]
                    g_opt_th = optim.Adam(th_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    d_mod = [CNN_Dis_Ter_Thres_gp(3)]
                    all_dparam_new = [param for name, param in d_mod[0].named_parameters() if not 'delta_th' in name]
                    th_dparam_new = [param for name, param in d_mod[0].named_parameters() if 'delta_th' in name]
                    d_opt_all = [optim.Adam(all_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                    d_opt_th = [optim.Adam(th_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]

                    gan = WGAN_GP_Thres(args, save_folder, g_mod, d_mod, [g_opt_all, g_opt_th],
                                        [d_opt_all, d_opt_th], fid_state='FID')
                else:
                    g_mod = CNN_Gen_gp(3)  # Full model
                    d_mod = [CNN_Dis_gp(3)]

                if instructions[k][1] != 'Ternarized_with_Threshold':
                    # Create optimizers for each WGAN model
                    g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                    d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate,
                                        betas=(args.beta_1, args.beta_2))]
                    # Create training object
                    gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')

                dataset, _ = get_data_loader_custom(args, args.batch_size, tr_cls, subsample=0)
                dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, tr_cls, subsample=0)
                # Train each pre-trained generator and discriminator models
                loc_string = gan.train(dataset, dataset_fid, discriminator_save_iter=-1)
                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'Barycenter':
            tr_cls = []
            d_mod = []
            d_opt = []
            for kk in range(len(instructions[k][2])):
                tr_cls.append(instructions[k][2][kk])
            save_folder = 'LSUN_Simulations/Folder_' + str(instructions[k][2])
            if os.path.exists(save_folder):
                # This model ha already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                # Perform Training
                for kk_ in range(len(instructions[k][2])):
                    d_mod.append(CNN_Dis_gp(3))
                    d_opt.append(optim.Adam(d_mod[kk_].parameters(), lr=args.learning_rate,
                                            betas=(args.beta_1, args.beta_2)))
                # Create temporary generator model and optimizers. These won't be used in training.
                g_mod = CNN_Gen_gp(3)     # Not used due to transfer learning
                g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
                # Create training object
                gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt)
                G_prev = []
                dataset = []
                for ind in range(len(instructions[k][2])):
                    path = args.train_classes[instructions[k][2][ind]]
                    if type(path) is str:
                        checkpoint = torch.load(path)
                        G_prev.append(checkpoint['modelG'])
                        dataset.append(gan.generate(gen=G_prev[ind], samp_num=args.sample_number, iter=0,
                                                    fld_str='LSUN_Simulations/data_folder'+str(instructions[k][3])
                                                            + str(ind)))
                    else:
                        G_prev.append(CNN_Gen_gp(3))
                        dataset_new, _ = get_data_loader_custom(args, args.batch_size, tr_cls[ind], subsample=0)
                        dataset.append(dataset_new)
                # Train each pre-trained generator and discriminator models
                G_prev_opt = optim.Adam(G_prev[0].parameters(), lr=args.learning_rate,
                                        betas=(args.beta_1, args.beta_2))
                gan_bary = WGAN_GP(args, save_folder, G_prev[0], d_mod, G_prev_opt, d_opt)
                loc_string = gan_bary.train(dataset, [], discriminator_save_iter=-1)
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'Ensemble':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample

            tr_cls = []
            for kk in range(1, len(instructions[k][2])):
                tr_cls.append(instructions[k][2][kk])
            save_folder = 'LSUN_Simulations/Folder_Aggregate_' + str(instructions[k][2]) + instructions[k][1] \
                          + str(instructions[k][3])
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training is completed afterwards
                # Perform Training
                d_mod = [CNN_Dis_gp(3)]
                d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                G_prev = []
                # Ensemble datasets
                dataset = [ensemble(args, instructions[k])]

                path = args.train_classes[instructions[k][2][0]]
                if type(path) is str:
                    checkpoint = torch.load(path)
                    G_prev.append(checkpoint['modelG'])
                else:
                    G_prev.append(CNN_Gen_gp(3))

                # Train each pre-trained generator and discriminator models
                G_prev_opt = optim.Adam(G_prev[0].parameters(), lr=args.learning_rate,
                                        betas=(args.beta_1, args.beta_2))
                gan_bary = WGAN_GP(args, save_folder, G_prev[0], d_mod, G_prev_opt, d_opt)
                # dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, tr_cls, subsample=0)
                loc_string = gan_bary.train(dataset, [], discriminator_save_iter=-1)
                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'Transfer':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample
            # Indices must refer to datasets!
            indices = instructions[k][2]
            tr_cls = []
            for it in range(len(indices)):
                tr_cls.append(args.train_classes[indices[it]])
            save_folder = 'LSUN_Simulations/Folder_Transfer_' + str(tr_cls) + instructions[k][1] \
                          + str(instructions[k][4])
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                # Perform Training
                path = args.train_classes[instructions[k][3]]  # Must be a path to a checkpoint file
                checkpoint = torch.load(path)
                G_prev = checkpoint['modelG']
                if instructions[k][1] == 'Ternarized_with_Threshold':
                    g_mod = CNN_Gen_Ter_Thres_gp(3)  # Ternarized model (previous model
                    # parameters will be transferred here.)
                    g_mod.load_state_dict(G_prev.state_dict(), strict=False)
                    all_gparam_new = [param for name, param in g_mod.named_parameters() if not 'delta_th' in name]
                    g_opt_all = optim.Adam(all_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    th_gparam_new = [param for name, param in g_mod.named_parameters() if 'delta_th' in name]
                    g_opt_th = optim.Adam(th_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    d_mod = [CNN_Dis_Ter_Thres_gp(3)]
                    all_dparam_new = [param for name, param in d_mod[0].named_parameters() if not 'delta_th' in name]
                    th_dparam_new = [param for name, param in d_mod[0].named_parameters() if 'delta_th' in name]
                    d_opt_all = [optim.Adam(all_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                    d_opt_th = [optim.Adam(th_dparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]

                    '''
                    when model is loaded with the pre-trained model, the original
                    initialized threshold are not correct anymore, which might be clipped
                    by the hard-tanh function.
                    '''
                    for name, module in g_mod.named_modules():
                        name = name.replace('.', '/')
                        class_name = str(module.__class__).split('.')[-1].split("'")[0]
                        if "quanConv2d" in class_name or "quanConvTranspose2d" in class_name:
                            module.delta_th.data = module.weight.abs().max() * module.init_factor.cuda()
                    # set the graident register hook to modify the gradient (gradient clipping)
                    for name, param in g_mod.named_parameters():
                        if "delta_th" in name:
                            # if "delta_th" in name and 'classifier' in name:
                            # based on previous experiment, the clamp interval would better range between 0.001
                            param.register_hook(lambda grad: grad.clamp(min=-0.001, max=0.001))

                    gan = WGAN_GP_Thres(args, save_folder, g_mod, d_mod, [g_opt_all, g_opt_th],
                                        [d_opt_all, d_opt_th], fid_state='FID')
                else:
                    g_mod = CNN_Gen_gp(3)  # Full model (previous model parameters will be transferred here.)
                    d_mod = [CNN_Dis_gp(3)]

                if instructions[k][1] != 'Ternarized_with_Threshold':
                    g_mod.load_state_dict(G_prev.state_dict())
                    # Create optimizers for each WGAN model
                    g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate,
                                       betas=(args.beta_1, args.beta_2))
                    d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate,
                                        betas=(args.beta_1, args.beta_2))]
                    # Create training object
                    # gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')

                path = args.train_classes[instructions[k][2][0]]
                if type(path) is str:
                    # Create training object
                    gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt)
                    checkpoint = torch.load(path)
                    data_G = checkpoint['modelG']
                    dataset = [gan.generate(gen=data_G, samp_num=args.sample_number, iter=0,
                                            fld_str='LSUN_Simulations/data_folder' + str(0)
                                                    + str(instructions[k][4]))]
                    dataset_fid = dataset
                else:
                    dataset, _ = get_data_loader_custom(args, args.batch_size, tr_cls, subsample=args.subsample)
                    dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, tr_cls, subsample=0)
                    # Create training object
                    gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')
                # Train each pre-trained generator and discriminator models
                loc_string = gan.train(dataset, dataset_fid, discriminator_save_iter=-1)
                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][4]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'BaryTransfer':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample
            # First data is barycenter data, remaining are target edge data
            ind0 = instructions[k][2][0]
            ind1 = instructions[k][2][1:]
            edge_cls = []
            for it in range(len(ind1)):
                edge_cls.append(args.train_classes[ind1[it]])
            bary_cls = [args.train_classes[ind0]]
            save_folder = 'LSUN_Simulations/Folder_BaryTransfer_' + str(edge_cls) + instructions[k][1] \
                          + str(instructions[k][4])
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                # Perform Training
                path = args.train_classes[instructions[k][3]]  # Must be a path to a checkpoint file
                checkpoint = torch.load(path)
                G_prev = checkpoint['modelG']
                if instructions[k][1] == 'Ternarized_with_Threshold':
                    g_mod = CNN_Gen_Ter_Thres_gp(3)  # Ternarized model (previous model
                    # parameters will be transferred here.)
                    g_mod.load_state_dict(G_prev.state_dict(), strict=False)
                    all_gparam_new = [param for name, param in g_mod.named_parameters() if not 'delta_th' in name]
                    g_opt_all = optim.Adam(all_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    th_gparam_new = [param for name, param in g_mod.named_parameters() if 'delta_th' in name]
                    g_opt_th = optim.Adam(th_gparam_new, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

                    d_mod = [CNN_Dis_Ter_Thres_gp(3), CNN_Dis_Ter_Thres_gp(3)]
                    all_dparam_new0 = [param for name, param in d_mod[0].named_parameters() if not 'delta_th' in name]
                    th_dparam_new0 = [param for name, param in d_mod[0].named_parameters() if 'delta_th' in name]
                    all_dparam_new1 = [param for name, param in d_mod[1].named_parameters() if not 'delta_th' in name]
                    th_dparam_new1 = [param for name, param in d_mod[1].named_parameters() if 'delta_th' in name]
                    d_opt_all = [optim.Adam(all_dparam_new0, lr=args.learning_rate, betas=(args.beta_1, args.beta_2)),
                                 optim.Adam(all_dparam_new1, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                    d_opt_th = [optim.Adam(th_dparam_new0, lr=args.learning_rate, betas=(args.beta_1, args.beta_2)),
                                optim.Adam(th_dparam_new1, lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]

                    '''
                    when model is loaded with the pre-trained model, the original
                    initialized threshold are not correct anymore, which might be clipped
                    by the hard-tanh function.
                    '''
                    for name, module in g_mod.named_modules():
                        name = name.replace('.', '/')
                        class_name = str(module.__class__).split('.')[-1].split("'")[0]
                        if "quanConv2d" in class_name or "quanConvTranspose2d" in class_name:
                            module.delta_th.data = module.weight.abs().max() * module.init_factor.cuda()
                    # set the graident register hook to modify the gradient (gradient clipping)
                    for name, param in g_mod.named_parameters():
                        if "delta_th" in name:
                            # if "delta_th" in name and 'classifier' in name:
                            # based on previous experiment, the clamp interval would better range between 0.001
                            param.register_hook(lambda grad: grad.clamp(min=-0.001, max=0.001))

                    gan = WGAN_GP_Thres(args, save_folder, g_mod, d_mod, [g_opt_all, g_opt_th],
                                        [d_opt_all, d_opt_th], fid_state='FID')
                else:
                    g_mod = CNN_Gen_gp(3)  # Full model (previous model parameters will be transferred here.)
                    d_mod = [CNN_Dis_gp(3), CNN_Dis_gp(3)]

                if instructions[k][1] != 'Ternarized_with_Threshold':
                    g_mod.load_state_dict(G_prev.state_dict())
                    # Create optimizers for each WGAN model
                    g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate,
                                       betas=(args.beta_1, args.beta_2))
                    d_opt = [optim.Adam(d_mod[0].parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2)),
                             optim.Adam(d_mod[1].parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))]
                    # Create training object
                    gan = WGAN_GP(args, save_folder, g_mod, d_mod, g_opt, d_opt, fid_state='FID')

                # ind0 must be a path
                dataset = []
                checkpoint = torch.load(bary_cls[0])
                G_prev_data = checkpoint['modelG']
                dataset.append(gan.generate(gen=G_prev_data, samp_num=args.sample_number, iter=0,
                                            fld_str='LSUN_Simulations/data_folder'+str(instructions[k][4])))
                dataset_new, _ = get_data_loader_custom(args, args.batch_size, edge_cls, subsample=args.subsample)
                dataset.append(dataset_new[0])
                dataset_fid, _ = get_data_loader_custom(args, args.FID_BATCH_SIZE, edge_cls, subsample=0)

                # Train each pre-trained generator and discriminator models
                loc_string = gan.train(dataset, dataset_fid, discriminator_save_iter=-1)
                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][4]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'ModelAverage':
            # Change batch_size if there is not enough samples!
            old_batch_size = args.batch_size
            if args.subsample != 0:
                if args.batch_size > args.subsample:
                    args.batch_size = args.subsample
            bary_cls = []
            ind = instructions[k][2]
            for it in range(len(ind)):
                bary_cls.append(args.train_classes[ind[it]])
            if len(bary_cls) > 25:
                save_folder = 'LSUN_Simulations/Folder_ModelAverage_Too_Long' \
                              + instructions[k][1] + str(instructions[k][3])
            else:
                save_folder = 'LSUN_Simulations/Folder_ModelAverage_' + instructions[k][1] \
                              + str(instructions[k][3])
            if os.path.exists(save_folder):
                # This model has already been trained. Skip to next item in args.train_classes
                k = k + 1
                continue
            else:
                os.mkdir(save_folder)  # Need to check if the training complete afterwards
                # Perform average operation
                G_model = []
                path = bary_cls[0]  # Must be a path to a checkpoint file
                checkpoint = torch.load(path)
                G_model.append(checkpoint['modelG'])
                params = list(G_model[0].parameters())
                for ell in range(len(bary_cls) - 1):
                    path = bary_cls[ell + 1]  # Must be a path to a checkpoint file
                    checkpoint = torch.load(path)
                    G_model.append(checkpoint['modelG'])
                    old_param = list(G_model[ell + 1].parameters())
                    for ll in range(len(params)):
                        params[ll] = params[ll] + old_param[ll]
                g_mod = CNN_Gen_gp(3).cuda(0)  # Full model (previous model parameters will be transferred here.)
                state_dict = g_mod.state_dict()
                layer_num = 0
                for name, param in state_dict.items():
                    # Transform the parameter as required.
                    transformed_param = params[layer_num] / len(bary_cls)
                    # Update the parameter.
                    if "weight" in name or "bias" in name:
                        state_dict[name].copy_(transformed_param)
                        layer_num += 1

                # state_dict = []
                # path = bary_cls[0]  # Must be a path to a checkpoint file
                # checkpoint = torch.load(path)
                # state_dict.append(checkpoint['modelG'].state_dict())
                # for ell in range(len(bary_cls)-1):
                #     path = bary_cls[ell + 1]  # Must be a path to a checkpoint file
                #     checkpoint = torch.load(path)
                #     state_dict.append(checkpoint['modelG'].state_dict())
                #
                #
                #     old_param = list(G_model[ell + 1].parameters())
                #     for ll in range(len(params)):
                #         params[ll] = params[ll] + old_param[ll]
                # # params = params / len(bary_cls)
                # g_mod = CNN_Gen_gp(3)  # Full model (previous model parameters will be transferred here.)
                # params_next = list(g_mod.parameters())
                # for mm in range(len(params_next)):
                #     temp = params[mm] / len(bary_cls)
                #     params_next[mm] = temp.clone().cuda(0)
                g_opt = optim.Adam(g_mod.parameters(), lr=args.learning_rate,
                                   betas=(args.beta_1, args.beta_2))
                check_file = save_folder + '/ModelAverage'
                torch.save({
                    'modelG': g_mod,
                    'optimizerG': g_opt
                }, check_file)
                loc_string = check_file

                # Reassign batch_size after complete
                args.batch_size = old_batch_size
                args.train_classes[instructions[k][3]] = loc_string
                with open('Classes.txt', 'w') as f:
                    for item in args.train_classes:
                        f.write("%s\n" % item)

        elif instructions[k][0] == 'subsample':
            args.subsample = int(instructions[k][1])
        elif instructions[k][0] == 'ensemble_sample_number':
            args.ensemble_sample_number = int(instructions[k][1])

        k = k + 1


if __name__ == '__main__':
    simulation_dir = 'LSUN_Simulations'
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)

    args = parse_args()
    print(args.cuda)
    main(args)
    print('Finished')