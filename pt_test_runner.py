# -*- coding: utf-8 -*-


"""
Usage example:

# SGD 
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-1 \
    --momentum 0.1 --num_epochs 50 --data_dir=/shared/datasets/DeepOBS &&
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-2 \
    --momentum 0.1 --num_epochs 50 --data_dir=/shared/datasets/DeepOBS &&
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-3 \
    --momentum 0.1 --num_epochs 50

# Adam
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-1 --num_epochs 50 &&
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-2 --num_epochs 50 &&
python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-3 --num_epochs 50


python pt_test_runner.py cifar100_allcnnc --bs 128 --lr 1e-3 --num_epochs 10 --data_dir=data_deepobs/cifar-100


Problems:
quadratic_deep, mnist_vae, fmnist_2c2d, cifar10_3c3d, fmnist_vae,
cifar100_allcnnc, cifar100_wrn164, cifar100_wrn404, svhn_3c3d,
svhn_wrn164, tolstoi_char_rnn, mnist_2c2d, mnist_mlp, fmnist_mlp,
mnist_logreg, fmnist_logreg

Sample result:


********************************
Evaluating after 10 of 10 epochs...
TRAIN: loss 3.54655
VALID: loss 3.54968
TEST: loss 3.55916
********************************
"""


import torch
import deepobs.pytorch as ptobs


optimizer_class = torch.optim.SGD  # (model.parameters(), lr=0.01, momentum=0.9)
hyperparams = {"lr": {"type": float},
               "momentum": {"type": float, "default": 0},
               "nesterov": {"type": bool, "default": False}}




# optimizer_class = torch.optim.Adam  # ([var1, var2], lr=0.0001)
# hyperparams = {"lr": {"type": float}}


runner = ptobs.runners.StandardRunner(optimizer_class, hyperparams)
# The run method accepts all the relevant inputs, all arguments that are not
# provided will automatically be grabbed from the command line.
runner.run(
    train_log_interval=10,
    # print_train_iter=True,
    # tb_log=True,
    # tb_log_dir="quack",
    skip_if_exists=False)


# import pdb; pdb.set_trace()


