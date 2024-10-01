import os, sys
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import train_eval as train_eval

# Add the cloned_repo directory to the system path
sys.path.insert(0, os.path.abspath('./ProtoPNet'))
from ProtoPNet.helpers import makedir
from ProtoPNet import model, push, save
from ProtoPNet.log import create_logger
from ProtoPNet.preprocess import mean, std, preprocess_input_function
from logger import WandbLogger
from config import base_architecture, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, train_dir, val_dir, train_push_dir, \
    train_batch_size, test_batch_size, train_push_batch_size, joint_optimizer_lrs, joint_lr_step_size, \
    warm_optimizer_lrs, last_layer_optimizer_lr, coefs, num_train_epochs, num_warm_epochs, push_start, push_epochs


def main():
    device = set_device()

    # book keeping namings and code

    model_dir = initialise_runs_dir()

    log, logclose = create_logger(
        log_filename=os.path.join(model_dir, 'train.log'))
    wandb_logger = WandbLogger({},
                               logger_name='ProtoPNet',
                               project='FinalProject')
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    train_loader, train_push_loader, val_loader = load_dataloaders()

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(val_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    protopnet = model.construct_PPNet(
        base_architecture=base_architecture,
        pretrained=True,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type)
    protopnet = protopnet.to(device)
    protopnet_parallel = torch.nn.DataParallel(protopnet)
    class_specific = True

    # define optimizer
    joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = initialise_optimisers(
        protopnet)

    # train the model
    log('start training')
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            train_eval.warm_only(model=protopnet_parallel, log=log)
            _ = train_eval.train(model=protopnet_parallel,
                                 dataloader=train_loader,
                                 optimizer=warm_optimizer,
                                 class_specific=class_specific,
                                 coefs=coefs,
                                 log=log,
                                 wandb_logger=wandb_logger)
        else:
            train_eval.joint(model=protopnet_parallel, log=log)
            joint_lr_scheduler.step()
            _ = train_eval.train(model=protopnet_parallel,
                                 dataloader=train_loader,
                                 optimizer=joint_optimizer,
                                 class_specific=class_specific,
                                 coefs=coefs,
                                 log=log,
                                 wandb_logger=wandb_logger)

        accu = train_eval.test(model=protopnet_parallel,
                               dataloader=val_loader,
                               class_specific=class_specific,
                               log=log,
                               wandb_logger=wandb_logger)
        save.save_model_w_condition(model=protopnet,
                                    model_dir=model_dir,
                                    model_name=str(epoch) + 'nopush',
                                    accu=accu,
                                    target_accu=0.5,
                                    log=log)

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader,
                prototype_network_parallel=protopnet_parallel,
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=
                prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=
                proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = train_eval.test(model=protopnet_parallel,
                                   dataloader=val_loader,
                                   class_specific=class_specific,
                                   log=log,
                                   wandb_logger=wandb_logger)
            save.save_model_w_condition(model=protopnet,
                                        model_dir=model_dir,
                                        model_name=str(epoch) + 'push',
                                        accu=accu,
                                        target_accu=0.5,
                                        log=log)

            train_eval.last_only(model=protopnet_parallel, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = train_eval.train(model=protopnet_parallel,
                                     dataloader=train_loader,
                                     optimizer=last_layer_optimizer,
                                     class_specific=class_specific,
                                     coefs=coefs,
                                     log=log,
                                     wandb_logger=wandb_logger)
                accu = train_eval.test(model=protopnet_parallel,
                                       dataloader=val_loader,
                                       class_specific=class_specific,
                                       log=log,
                                       wandb_logger=wandb_logger)
                save.save_model_w_condition(model=protopnet,
                                            model_dir=model_dir,
                                            model_name=str(epoch) + '_' +
                                            str(i) + 'push',
                                            accu=accu,
                                            target_accu=0.5,
                                            log=log)

    logclose()


def initialise_optimisers(ppnet):
    joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
         {'params': ppnet.add_on_layers.parameters(
         ), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
            {'params': ppnet.prototype_vectors,
                'lr': joint_optimizer_lrs['prototype_vectors']},
         ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors,
             'lr': warm_optimizer_lrs['prototype_vectors']},
         ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{
        'params': ppnet.last_layer.parameters(),
        'lr': last_layer_optimizer_lr
    }]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    return joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer


def load_dataloaders():
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset,
        batch_size=train_push_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    # test set
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=test_batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=False)

    return train_loader, train_push_loader, val_loader


def initialise_runs_dir():
    runs_dir = './saved_models/' + base_architecture + '/'
    makedir(runs_dir)

    from config import experiment_run
    if not experiment_run:
        latest_run = 0
        for dir in os.listdir(runs_dir):
            dir_path = os.path.join(runs_dir, dir)
            if os.path.isdir(dir_path):
                try:
                    dir_int = int(dir)
                    if dir_int > latest_run:
                        latest_run = dir_int
                except ValueError:
                    continue

        experiment_run = f'{latest_run + 1}'
    model_dir = runs_dir + experiment_run + '/'
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'config.py'), dst=model_dir)
    return model_dir


def set_device():
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'device: {device}')
    return device


if __name__ == '__main__':
    main()
