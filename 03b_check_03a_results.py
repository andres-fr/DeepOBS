#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Assuming runs were properly generated on 03a, this script checks the
final performances achieved. Its goal is to confirm that we were able to
reproduce the results from the benchmark paper, before we move onto


python 03b_check_03a_results.py
"""


import os
#
from deepobs.analyzer import get_performance_dictionary


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################

if __name__ == "__main__":


    MODE = "most"
    METRIC = "valid_accuracies"
    #
    results_path = "results"
    adam_c100_path = os.path.join("results", "cifar100_allcnnc", "Adam")
    adam_c10_path = os.path.join("results", "cifar10_3c3d", "Adam")
    converg_path = os.path.join("baselines_deepobs",
                                "convergence_performance.json")

    adam_c100_perf = get_performance_dictionary(
        adam_c100_path, mode=MODE, metric=METRIC, conv_perf_file=converg_path)
    adam_c10_perf = get_performance_dictionary(
        adam_c10_path, mode=MODE, metric=METRIC, conv_perf_file=converg_path)

    print("Final", METRIC)
    print("Adam on CIFAR100:", adam_c100_perf["Performance"])
    print("Adam on CIFAR10:", adam_c10_perf["Performance"])

    # import pdb; pdb.set_trace()
