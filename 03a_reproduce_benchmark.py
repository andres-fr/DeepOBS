#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Now:

wipe our results. then use the scheduled runner with the given params to reproduce valley results
https://deepobs.readthedocs.io/en/develop/_modules/deepobs/pytorch/runners/runner.html#LearningRateScheduleRunner

We should get similar metrics and loss progressions,
this can be checked by comparing the lines via following fn

# variant of the above, in which a given optimizer is compared to
# another reference optimizer
fig2b, ax2b = plot_optimizer_performance(
    os.path.join(problem_path, "AdamOptimizer"),
    mode=MODE,
    metric=METRIC,
    reference_path=os.path.join(problem_path,
                                "GradientDescentOptimizer"),
    show=SHOW_PLOTS,
    which="mean_and_std",
    yscale_loss="log",
    yscale_acc="logit")
"""


import os
#
from deepobs import CrowdedValleyPaths
import deepobs.analyzer
from deepobs.analyzer import plot_optimizer_performance

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



if __name__ == "__main__":
    import pdb; pdb.set_trace()
