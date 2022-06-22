#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
python analyzer_api_test.py

Change the hardcoded paths and the 'if 0' to 'if 1' to test the desired
functionality.
"""


import os
from unittest.mock import patch
#
import torch
import pandas as pd
#
import deepobs
from deepobs.analyzer import (
    check_output,
    estimate_runtime,
    get_performance_dictionary,
    plot_final_metric_vs_tuning_rank,
    plot_hyperparameter_sensitivity,
    plot_hyperparameter_sensitivity_2d,
    plot_optimizer_performance,
    plot_results_table,
    plot_testset_performances)
from deepobs.analyzer.shared_utils import create_setting_analyzer_ranking


if __name__ == "__main__":

    # hardcoded global paths to benchmark results
    benchmark_path = os.path.join(
        os.path.expanduser("~"), "git-work", "Crowded-Valley---Results",
        "results_main", "medium_budget", "none")
    problem_path = os.path.join(benchmark_path, "cifar10_3c3d")
    adam_path = os.path.join(problem_path, "AdamOptimizer")
    conv_perf_path = os.path.join("baselines_deepobs",
                                  "convergence_performance.json")


    if 0:
        # Sanity check for the OBS results
        check_output(benchmark_path)


    if 0:
        # Here we emulate a parameterless call because otherwise the runner
        # will try to make sense of this script's arguments, and fail.
        # All necessary arguments for the runner are passed as fn params.
        with patch("sys.argv", [__file__]):
            runtime_analysis = estimate_runtime(
                "pytorch",
                deepobs.pytorch.runners.StandardRunner, torch.optim.Adam,
                optimizer_hp={"lr": {"type": float}},
                optimizer_hyperparams={"lr": 0.001},
                n_runs=3, sgd_lr=0.01, testproblem="mnist_mlp",
                num_epochs=3, batch_size=128)
            print(runtime_analysis)


    if 0:
        # mode can be:
        # * final: Settings are sorted by get_final_value
        # * best: Settings are sorted by get_best_value
        # * most: Sorted by (descending) number of seeds, or final if all equal
        # Then, get_performance_dictionary returns the stats for the "best",
        # where "Performance" is again given by the "mode" criterion
        #
        # Therefore, the protocol from the benchmark paper can be implemented
        # as follows: after an initial hyperparameter search, "best" can be
        # used to find the best "setting", then that setting can be run
        # multiple times, and then "most" can be used
        perf_dict = get_performance_dictionary(
            adam_path, mode="final", metric="valid_accuracies",
            conv_perf_file=conv_perf_path)

        # This is the auxiliar of get_performance_dictionary, returning the
        # hyperparametrizations sorted by the mode.
        setting_analyzer_ranking = create_setting_analyzer_ranking(
            adam_path, mode="best", metric="valid_accuracies")[0]


    if 0:
        # this doesn't plot anything, returns a dataframe with rows=tasks, and
        # columns=optimizers. For each task+optimizer, the table contains the
        # training params, opt hyperpars, performance (as per "mode"), and
        # speed (if conv_perf_path is given)
        table = plot_results_table(
            benchmark_path, mode="most", metric="valid_accuracies",
            conv_perf_file=conv_perf_path)
        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None,
                               "display.max_colwidth", None,
                               "display.width", 300):
            print(table.loc[:, table.columns[0]])
        print(table.columns)


    if 0:
        # render different plots
        SHOW_PLOTS = False

        fig1, ax1 = plot_final_metric_vs_tuning_rank(
            adam_path, metric="valid_accuracies", show=SHOW_PLOTS)

        fig2, ax2 = plot_optimizer_performance(
            problem_path,
            mode="most",
            metric="valid_accuracies",
            reference_path=None,
            show=SHOW_PLOTS,
            which="mean_and_std")

        fig3, ax3 = plot_testset_performances(
            benchmark_path,
            mode="most",
            metric="valid_accuracies",
            reference_path=None,
            show=SHOW_PLOTS,
            which="mean_and_std")

        fig4, ax4 = plot_hyperparameter_sensitivity_2d(
            adam_path,
            ("learning_rate", "beta1"),
            mode="final",
            metric="valid_accuracies",
            xscale="log",
            yscale="linear",
            show=SHOW_PLOTS)

        fig5, ax5 = plot_hyperparameter_sensitivity(
            problem_path,
            "learning_rate",
            mode="final",
            metric="valid_accuracies",
            xscale="log",
            plot_std=True,
            reference_path=None,
            show=SHOW_PLOTS)


    import pdb; pdb.set_trace()
