base_architecture = 'vgg19'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

import socket
hostname = socket.gethostname()
if hostname.endswith('local'):  # Example check for local machine names
    print("Running on Macbook locally")
    data_path = '/Users/youssefshaarawy/Documents/Datasets/JustRAIGS'
else:
    print(f"Running on remote server: {hostname}")
    data_path = '/users/adfx751/Datasets/JustRAIGS'


train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
