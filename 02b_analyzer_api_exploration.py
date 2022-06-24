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
        best_setting = create_setting_analyzer_ranking(
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

    if 1:
        # render different plots. They can be saved with e.g.
        # fig.savefig(path, dpi=fig.dpi) or displayed with fig.show()
        SHOW_PLOTS = False
        # some plots use "generate_tuning_summary", which
        MODE = "most"  # doesn't implement the "most" mode
        METRIC = "valid_accuracies"

        # Monot. descending line plot, where each x-position is a unique
        # hyperparametrization and the y-axis is the accuracy. Note that a
        # hyperpar can have multiple seeds, they will show up on the same
        # vert line, giving an idea of the noise in accuracy due to seed.
        # When considering a single opt, this can be used to visualize
        # the best setting in its context
        fig1, ax1 = plot_final_metric_vs_tuning_rank(
            adam_path, metric="valid_accuracies", show=SHOW_PLOTS)

        # 4 line plots: x-axis is always training epoch, and each y-axis
        # represents one of {train, test}*{loss, accuracy}.
        # Each plot contains one line per optimizer, and this line can be
        # surrounded by a shadow representing e.g. "mean_and_std".
        # generally we see that loss goes down and acc goes up.
        #
        # Note that the path can be a single optimizer, or a full problem,
        # in which case it can be used to compare different optimizers.
        fig2a, ax2a = plot_optimizer_performance(
            problem_path,
            mode=MODE,
            metric=METRIC,
            reference_path=None,
            show=SHOW_PLOTS,
            which="mean_and_std",
            yscale_loss="log",
            yscale_acc="logit")

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

        # This function returns a matrix of plots, with 4 rows representing
        # the {train, test}*{loss, accuracy} values on the y-axis, and one
        # column per problem (e.g. quad_deep), and the x-axis representing
        # training epoch.
        # Note that the path and ref_path need to contain multiple tasks,
        # so if you want to use this to compare specific reference optimizers
        # you need to prepare a "results" folder with only those optimizers.
        fig3, ax3 = plot_testset_performances(
            benchmark_path,
            mode=MODE,
            metric=METRIC,
            reference_path=None,
            show=SHOW_PLOTS,
            which="mean_and_std")

        # Given a specific optimizer+task that has been run on multiple
        # settings, We can check the "metric" (e.g. valid_acc) as a function
        # of 2 hyperparameters with this plot. This can be helpful to
        # identify hyperpar correlations and other compound trends, although
        # the 1d version may be better to identify dominant hyperpars.
        fig4, ax4 = plot_hyperparameter_sensitivity_2d(
            adam_path,
            ("learning_rate", "beta1"),  # x and y axes.
            mode="final",
            metric=METRIC,  # given by color intensity
            xscale="log",
            yscale="linear",
            show=SHOW_PLOTS)

        # 1D version of the above, where the x axis is the given hyperpar,
        # and y is the given metric.. Since we have more space, we can show
        # here multiple optimizers by passing a problem_path instead of an
        # optimizer_path. This may get crowded though, so the recommended
        # way is to compare one (or a few) against a reference, as follows:
        fig5, ax5 = plot_hyperparameter_sensitivity(
            adam_path,
            "learning_rate",
            mode="final",
            metric=METRIC,
            xscale="log",
            plot_std=True,
            reference_path=os.path.join(problem_path,
                                        "GradientDescentOptimizer"),
            show=SHOW_PLOTS)

    import pdb; pdb.set_trace()
