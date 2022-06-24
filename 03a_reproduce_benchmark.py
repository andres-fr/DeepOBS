#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This parameterless script can be run e.g. in SLURM via:

python -u 03a_reproduce_benchmark.py

As usual, output will be generated into the ./results directory.
"""


import torch
import argparse
#
from deepobs import pytorch as ptobs


# ##############################################################################
# # ARGPARSE
# ##############################################################################

# We require an integer seed because otherways 42 is always picked
argparser = argparse.ArgumentParser(description="Run DeepOBS")
argparser.add_argument("-s","--seed", required=True, type=int,
                       help="Specify an integer seed for the DeepOBS runner")
args = parser.parse_args()
SEED = args.seed

import pdb; pdb.set_trace()

# ##############################################################################
# # GLOBALS
# ##############################################################################

# these are hardcoded, see paper
C100_BATCH_SIZE = 256
C100_EPOCHS = 350
C10_BATCH_SIZE = 128
C10_EPOCHS = 100


# This adam+large+cosine setting gave a "most"-best CIFAR100 valid accuracy
# of 0.578175
C100_BEST = {"lrsched": "ltr",
             "Performance": 0.5781750801282051,
             "Hyperparameters": {"learning_rate": 0.0016536937182824417,
                                 "beta1": 0.5350043427572003,
                                 "beta2": 0.9313393657785078,
                                 "epsilon": 1e-08,
                                 "use_locking": False},
             "Training Parameters": {"lr_sched_epochs": [
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
                 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
                 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
                 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
                 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
                 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
                 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303,
                 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315,
                 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
                 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339,
                 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350],
                                     "lr_sched_factors": [
                 0.02857142857142857, 0.05714285714285714, 0.08571428571428572,
                 0.11428571428571428, 0.14285714285714285, 0.17142857142857143,
                 0.2, 0.22857142857142856, 0.2571428571428571,
                 0.2857142857142857, 0.3142857142857143, 0.34285714285714286,
                 0.37142857142857144, 0.4, 0.42857142857142855,
                 0.45714285714285713, 0.4857142857142857, 0.5142857142857142,
                 0.5428571428571428, 0.5714285714285714, 0.6,
                 0.6285714285714286, 0.6571428571428571, 0.6857142857142857,
                 0.7142857142857143, 0.7428571428571429, 0.7714285714285715,
                 0.8, 0.8285714285714286, 0.8571428571428571,
                 0.8857142857142857, 0.9142857142857143, 0.9428571428571428,
                 0.9714285714285714, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 0.9722222222222222, 0.9444444444444444, 0.9166666666666666,
                 0.8888888888888888, 0.8611111111111112, 0.8333333333333334,
                 0.8055555555555556, 0.7777777777777778, 0.75,
                 0.7222222222222222, 0.6944444444444444, 0.6666666666666667,
                 0.6388888888888888, 0.6111111111111112, 0.5833333333333333,
                 0.5555555555555556, 0.5277777777777778, 0.5,
                 0.4722222222222222, 0.4444444444444444, 0.41666666666666663,
                 0.38888888888888884, 0.36111111111111116, 0.33333333333333337,
                 0.3055555555555556, 0.2777777777777778, 0.25,
                 0.2222222222222222, 0.19444444444444442, 0.16666666666666663,
                 0.13888888888888884, 0.11111111111111116, 0.08333333333333337,
                 0.05555555555555558, 0.02777777777777779]}}

c100_best_params = {"lr": 0.0016536937182824417,
                    "betas": (0.5350043427572003, 0.9313393657785078),
                    "eps": 1e-08}

# This adam+large+cosine setting gave a "most"-best CIFAR10 valid accuracy
# of 0.94
C10_BEST = {"lrsched": "cosine",
            "Performance": 0.9400540865384615,
            "Speed": 39.7,
            "Hyperparameters": {"learning_rate": 0.0005853980163494224,
                                "beta1": 0.5777455997736856,
                                "beta2": 0.9806512476297935,
                                "epsilon": 1e-08,
                                "use_locking": False},
            "Training Parameters": {"lr_sched_epochs": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100],
                                    "lr_sched_factors": [
                0.9997532801828658, 0.9990133642141358, 0.99778098230154,
                0.996057350657239, 0.9938441702975689, 0.9911436253643444,
                0.9879583809693737, 0.9842915805643155, 0.9801468428384715,
                0.9755282581475768, 0.9704403844771128, 0.9648882429441257,
                0.9588773128419905, 0.9524135262330098, 0.9455032620941839,
                0.9381533400219317, 0.9303710135019718, 0.9221639627510075,
                0.9135402871372809, 0.9045084971874737, 0.8950775061878452,
                0.8852566213878946, 0.8750555348152298, 0.8644843137107058,
                0.8535533905932737, 0.8422735529643444, 0.8306559326618259,
                0.8187119948743449, 0.8064535268264883, 0.7938926261462366,
                0.7810416889260653, 0.7679133974894983, 0.7545207078751857,
                0.7408768370508576, 0.7269952498697734, 0.7128896457825363,
                0.6985739453173903, 0.6840622763423391, 0.6693689601226458,
                0.6545084971874737, 0.6394955530196147, 0.6243449435824275,
                0.6090716206982714, 0.5936906572928624, 0.5782172325201155,
                0.5626666167821521, 0.5470541566592573, 0.5313952597646567,
                0.5157053795390641, 0.5, 0.48429462046093585,
                0.4686047402353433, 0.45294584334074284, 0.4373333832178478,
                0.42178276747988447, 0.4063093427071376, 0.39092837930172886,
                0.3756550564175727, 0.3605044469803854, 0.34549150281252633,
                0.3306310398773543, 0.31593772365766104, 0.30142605468260975,
                0.28711035421746367, 0.2730047501302266, 0.2591231629491423,
                0.24547929212481434, 0.23208660251050156, 0.21895831107393482,
                0.2061073738537635, 0.19354647317351187, 0.18128800512565513,
                0.16934406733817414, 0.15772644703565564, 0.14644660940672627,
                0.13551568628929433, 0.1249444651847702, 0.11474337861210543,
                0.1049224938121548, 0.09549150281252633, 0.08645971286271903,
                0.07783603724899257, 0.06962898649802823, 0.06184665997806832,
                0.054496737905816106, 0.04758647376699032, 0.04112268715800943,
                0.035111757055874326, 0.029559615522887273,
                0.024471741852423234, 0.019853157161528467,
                0.015708419435684462, 0.012041619030626283,
                0.008856374635655695, 0.00615582970243117,
                0.0039426493427611176, 0.002219017698460002,
                0.0009866357858642205, 0.0002467198171342, 0.0]}}

c10_best_params = {"lr": 0.0005853980163494224,
                   "betas": (0.5777455997736856, 0.9806512476297935),
                   "eps": 1e-08}


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################

if __name__ == "__main__":

    # elements common to both runs
    optimizer_class = torch.optim.Adam  # ([var1, var2], lr=0.0001)
    hyperparams = {"lr": {"type": float},
                   "betas": {"type": tuple},
                   "eps": {"type": float}}
    runner = ptobs.runners.LearningRateScheduleRunner(
        optimizer_class, hyperparams)

    # run the CIFAR10 best setting with some random seed
    runner.run(
        testproblem="cifar10_3c3d",  # cifar100_allcnnc
        hyperparams=c10_best_params,
        batch_size=C10_BATCH_SIZE,
        num_epochs=C10_EPOCHS,
        train_log_interval=10,
        lr_sched_epochs=C10_BEST["Training Parameters"]["lr_sched_epochs"],
        lr_sched_factors=C10_BEST["Training Parameters"]["lr_sched_factors"],
        # print_train_iter=True,
        # tb_log=True,
        # tb_log_dir="quack",
        skip_if_exists=False)

    # run the CIFAR100 best setting with some random seed
    runner.run(
        testproblem="cifar100_allcnnc",
        hyperparams=c100_best_params,
        batch_size=C100_BATCH_SIZE,
        num_epochs=C100_EPOCHS,
        train_log_interval=10,
        lr_sched_epochs=C100_BEST["Training Parameters"]["lr_sched_epochs"],
        lr_sched_factors=C100_BEST["Training Parameters"]["lr_sched_factors"],
        # print_train_iter=True,
        # tb_log=True,
        # tb_log_dir="quack",
        skip_if_exists=False)

    # import pdb; pdb.set_trace()
