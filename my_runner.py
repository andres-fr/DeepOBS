# -*- coding: utf-8 -*-


"""
"""


# ##############################################################################
# # TENSORFLOW
# ##############################################################################


# import tensorflow as tf
# import deepobs.tensorflow as tfobs


# # optimizer_class = tf.train.MomentumOptimizer
# # hyperparams = [{"name": "momentum", "type": float},
# #                {"name": "use_nesterov", "type": bool, "default": False }]

# optimizer_class = tf.compat.v1.train.MomentumOptimizer
# hyperparams = {"momentum": {"type": float},
#                "learning_rate": {"type": float},
#                "use_nesterov": {"type": bool, "default": False}}


# runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)
# # The run method accepts all the relevant inputs, all arguments that are not
# # provided will automatically be grabbed from the command line.
# runner.run(train_log_interval=10)



# # import pdb; pdb.set_trace()




# ##############################################################################
# # PYTORCH
# # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
# ##############################################################################


import torch
import deepobs.pytorch as ptobs

# ptobs: from .runner import LearningRateScheduleRunner, PTRunner, StandardRunner

# optimizer_class = tf.train.MomentumOptimizer
# hyperparams = [{"name": "momentum", "type": float},
#                {"name": "use_nesterov", "type": bool, "default": False }]


optimizer_class = torch.optim.SGD  # (model.parameters(), lr=0.01, momentum=0.9)
# optimizer_class = torch.optim.Adam  # ([var1, var2], lr=0.0001)

hyperparams = {"lr": {"type": float},
               "momentum": {"type": float, "default": 0},
               "nesterov": {"type": bool, "default": False}}


runner = ptobs.runners.StandardRunner(optimizer_class, hyperparams)
# The run method accepts all the relevant inputs, all arguments that are not
# provided will automatically be grabbed from the command line.
runner.run(
    train_log_interval=1,
    print_train_iter=True,
    tb_log=True,
    tb_log_dir="quack",
    skip_if_exists=False)



# import pdb; pdb.set_trace()
