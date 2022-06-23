#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""


import os
# import torch

# #
# import deepobs
# from deepobs.analyzer import (
#     check_output,
#     estimate_runtime,
#     get_performance_dictionary,
#     plot_final_metric_vs_tuning_rank,
#     plot_hyperparameter_sensitivity,
#     plot_hyperparameter_sensitivity_2d,
#     plot_optimizer_performance,
#     plot_results_table,
#     plot_testset_performances)
# from deepobs.analyzer.shared_utils import create_setting_analyzer_ranking

import deepobs.analyzer
from deepobs import CrowdedValleyPaths
from deepobs.analyzer import get_performance_dictionary
from deepobs.analyzer import plot_final_metric_vs_tuning_rank

# if __name__ == "__main__":

# # hardcoded global paths to benchmark results
# large_budget_path = os.path.join(
#     os.path.expanduser("~"), "git-work", "Crowded-Valley---Results",
#     "results_main", "large_budget")
# tasks_none_path = os.path.join(large_budget_path, none)
# tasks_none_path = os.path.join(large_budget_path, none)
# tasks_none_path = os.path.join(large_budget_path, none)
# tasks_none_path = os.path.join(large_budget_path, none)


# figure global paths
conv_perf_path = os.path.join("baselines_deepobs",
                              "convergence_performance.json")

pather = deepobs.CrowdedValleyPaths(os.path.join(
    os.path.expanduser("~"), "git-work",
    "Crowded-Valley---Results", "results_main"))


# we focus on Adam for 2 CIFAR tasks
adam_c100_paths = pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
                         tasks={"cifar100_allcnnc"}, abspaths=False)
adam_c10_paths = pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
                        tasks={"cifar10_3c3d"}, abspaths=False)


# extract best result for each lr schedule, using the "most" criterion
adam_c100_perfs = [{"lrsched": lrsch, **get_performance_dictionary(
    p, mode="most", metric="valid_accuracies", conv_perf_file=conv_perf_path)}
                   for lrsch, p
                   in adam_c100_paths[["lrsched", "abspath"]].values.tolist()]
adam_c10_perfs = [{"lrsched": lrsch, **get_performance_dictionary(
    p, mode="most", metric="valid_accuracies", conv_perf_file=conv_perf_path)}
                  for lrsch, p
                  in adam_c10_paths[["lrsched", "abspath"]].values.tolist()]
best_adam_100 = max(adam_c100_perfs, key=lambda elt: elt["Performance"])
best_adam_10 = max(adam_c10_perfs, key=lambda elt: elt["Performance"])


# double-check that provided Performance is not noisy
print("Best adam 100 performance:", best_adam_100["Performance"])
plot_final_metric_vs_tuning_rank(
    # the path to the best adam we found
    pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
           tasks={"cifar100_allcnnc"}, lrscheds={best_adam_100["lrsched"]},
           abspaths=True)[0], metric="valid_accuracies", show=True)

print("Best adam 10 performance:", best_adam_10["Performance"])
plot_final_metric_vs_tuning_rank(
    # the path to the best adam we found
    pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
           tasks={"cifar10_3c3d"}, lrscheds={best_adam_10["lrsched"]},
           abspaths=True)[0], metric="valid_accuracies", show=True)



import pdb; pdb.set_trace()

# benchmark_path = os.path.join(
#     os.path.expanduser("~"), "git-work", "Crowded-Valley---Results",
#     "results_main", "large_budget", "none")
# problem_path = os.path.join(benchmark_path, "cifar10_3c3d")
# adam_path = os.path.join(problem_path, "AdamOptimizer")
# conv_perf_path = os.path.join("baselines_deepobs",
#                               "convergence_performance.json")



import pdb; pdb.set_trace()



"""
plot is test,

but "most" is problematic because there is no criterion??? so probably test is given???




plot the 3 curves, valid, test, train_val


TODO:

1. the dict function is not picking the right metric, fix that
2. Add 3 curves to the plot function
3. Check that no inconsistencies anymore
"""
