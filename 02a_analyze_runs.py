#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script browses the Crowded Valley results for a specific set of tasks
and optimizers, and retrieves the best results, with their corresponding
hyperparametrizations.

It also provides some sanity checks along the way.
"""


import os
#
from deepobs import CrowdedValleyPaths
import deepobs.analyzer
from deepobs.analyzer import get_performance_dictionary
from deepobs.analyzer import plot_final_metric_vs_tuning_rank


if __name__ == "__main__":

    METRIC = "test_accuracies"  # train/valid/test
    # figure global paths
    converg_path = os.path.join("baselines_deepobs",
                                  "convergence_performance.json")

    pather = deepobs.CrowdedValleyPaths(os.path.join(
        os.path.expanduser("~"), "git-work",
        "Crowded-Valley---Results", "results_main"))

    # we focus on Adam for 2 CIFAR tasks
    adam_c100_paths = pather(
        budgets={"large_budget"}, optimizers={"AdamOptimizer"},
        tasks={"cifar100_allcnnc"}, abspaths=False)
    adam_c10_paths = pather(
        budgets={"large_budget"}, optimizers={"AdamOptimizer"},
        tasks={"cifar10_3c3d"}, abspaths=False)

    # extract best result for each lr schedule, using the "most" criterion
    adam_c100_perfs = [{"lrsched": lrsch, **get_performance_dictionary(
        p, mode="most", metric=METRIC, conv_perf_file=converg_path)}
                       for lrsch, p in adam_c100_paths[
                               ["lrsched", "abspath"]].values.tolist()]
    adam_c10_perfs = [{"lrsched": lrsch, **get_performance_dictionary(
        p, mode="most", metric=METRIC, conv_perf_file=converg_path)}
                      for lrsch, p in adam_c10_paths[
                              ["lrsched", "abspath"]].values.tolist()]
    best_adam_100 = max(adam_c100_perfs, key=lambda elt: elt["Performance"])
    best_adam_10 = max(adam_c10_perfs, key=lambda elt: elt["Performance"])

    # double check that provided Performance isn't noisy, i.e. that the setting
    # with most runs has a good average equal to our "best", and with small
    # stddev. This way we know that our target Perf is good *and* reliable
    print(f"Best adam 100 performance ({METRIC, best_adam_10['lrsched']}):",
          best_adam_100["Performance"])
    plot_final_metric_vs_tuning_rank(
        # the path to the best adam we found
        pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
               tasks={"cifar100_allcnnc"}, lrscheds={best_adam_100["lrsched"]},
               abspaths=True)[0], metric=METRIC, show=True)
    #
    print(f"Best adam 10 performance ({METRIC, best_adam_10['lrsched']}):",
          best_adam_10["Performance"])
    plot_final_metric_vs_tuning_rank(
        # the path to the best adam we found
        pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
               tasks={"cifar10_3c3d"}, lrscheds={best_adam_10["lrsched"]},
               abspaths=True)[0], metric=METRIC, show=True)

    # Report final results
    print("\n\nRESULTS (Adam+large budget):")
    print("\n\ncifar100_allcnnc:")
    print(best_adam_100)
    print("\n\ncifar10_3c3d:")
    print(best_adam_10)
