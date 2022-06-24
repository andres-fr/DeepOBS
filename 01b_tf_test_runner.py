# -*- coding: utf-8 -*-


"""
Usage example:
python tf_test_runner.py quadratic_deep --bs 128 --learning_rate 1e-2 \
    --momentum 0.99 --num_epochs 10

Sample result:

********************************
Evaluating after 10 of 10 epochs...
TRAIN: loss 152.87
VALID: loss 154.935
TEST: loss 155.575
********************************
"""


import tensorflow as tf
import deepobs.tensorflow as tfobs


optimizer_class = tf.compat.v1.train.MomentumOptimizer
hyperparams = {"momentum": {"type": float},
               "learning_rate": {"type": float},
               "use_nesterov": {"type": bool, "default": False}}


runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)
# The run method accepts all the relevant inputs, all arguments that are not
# provided will automatically be grabbed from the command line.
runner.run(train_log_interval=10)


# import pdb; pdb.set_trace()
