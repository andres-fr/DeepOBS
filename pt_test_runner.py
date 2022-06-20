# -*- coding: utf-8 -*-


"""
Usage example:

# SGD
python pt_test_runner.py quadratic_deep --bs 128 --lr 1e-2 \
    --momentum 0.99 --num_epochs 100


# Adam
python pt_test_runner.py quadratic_deep --bs 128 --lr 1e-2 --num_epochs 100



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


