#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
python plot_results.py --baseline_path=${WORK}/model_snapshots/DeepOBS results/

python plot_results.py --baseline_path=${WORK}/model_snapshots/DeepOBS \
     ${WORK}/Crowded-Valley---Results/results_main


python plot_results.py --baseline_path=${WORK}/model_snapshots/DeepOBS \
     ${HOME}/git-work/Crowded-Valley---Results/results_main


"""


from __future__ import print_function
import argparse
#
import deepobs


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting tool for DeepOBS.")
    parser.add_argument("path", help="Path to the results folder")
    parser.add_argument(
        "--get_best_run",
        action="store_const",
        const=True,
        default=False,
        help="Return best hyperparameter setting per optimizer and testproblem.",
    )
    parser.add_argument(
        "--plot_lr_sensitivity",
        action="store_const",
        const=True,
        default=False,
        help="Plot 'sensitivity' plot for the learning rates.",
    )
    parser.add_argument(
        "--plot_performance",
        action="store_const",
        const=True,
        default=False,
        help="Plot performance plot compared to the baselines.",
    )
    parser.add_argument(
        "--plot_table",
        action="store_const",
        const=True,
        default=False,
        help="Plot overall performance table including speed and hyperparameters.",
    )
    parser.add_argument(
        "--full",
        action="store_const",
        const=True,
        default=False,
        help="Run a full analysis and plot all figures.",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default="baselines_deepobs",
        help="Path to baseline folder.",
    )
    return parser


def read_args():
    parser = parse_args()
    args = parser.parse_args()
    return args


def main(
    path,
    get_best_run,
    plot_lr_sensitivity,
    plot_performance,
    plot_table,
    full,
    baseline_path,
):
    # Put all input arguments back into an args variable, so I can use it as
    # before (without the main function)
    args = argparse.Namespace(**locals())
    # Parse whole baseline folder




    import os
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



    benchmark_path = os.path.join(
        os.path.expanduser("~"), "git-work", "Crowded-Valley---Results",
        "results_main", "medium_budget", "none")
    problem_path = os.path.join(benchmark_path, "cifar10_3c3d")
    adam_path = os.path.join(problem_path, "AdamOptimizer")
    conv_perf_path = os.path.join("baselines_deepobs", "convergence_performance.json")


    # check_output(benchmark_path)

    import pdb; pdb.set_trace()

    # estimate_runtime(
    #     framework, runner_cls, optimizer_cls,
    #     optimizer_hp, optimizer_hyperparams,
    #     n_runs=5, sgd_lr=0.01, testproblem="mnist_mlp",
    #     num_epochs=5, batch_size=128)

    perf_dict = get_performance_dictionary(
        adam_path, mode="best", metric="valid_accuracies", conv_perf_file=None)

    fig, ax = plot_final_metric_vs_tuning_rank(
        adam_path, metric="valid_accuracies", show=False)




    # This returns a descending list of
    setting_analyzer_ranking = create_setting_analyzer_ranking(
        adam_path, mode="best", metric="valid_accuracies")[0]
    best_adam = setting_analyzer_ranking[0]





    # plot_hyperparameter_sensitivity_2d(
    #     adam_path,
    #     ("learning_rate", "beta1"),
    #     mode="final",
    #     metric="valid_accuracies",
    #     xscale="log",
    #     yscale="linear",
    #     show=False)

    # plot_hyperparameter_sensitivity(
    #     problem_path,
    #     "learning_rate",
    #     mode="final",
    #     metric="valid_accuracies",
    #     xscale="log",
    #     plot_std=True,
    #     reference_path=None,
    #     show=False)


    plot_optimizer_performance(
        problem_path,
        mode="most",
        metric="valid_accuracies",
        reference_path=None,
        show=False,
        which="mean_and_std")


    # this doesn't plot anything, returns a dataframe
    table = plot_results_table(
        benchmark_path, mode="most", metric="valid_accuracies", conv_perf_file=None)


    plot_testset_performances(
        benchmark_path,
        mode="most",
        metric="valid_accuracies",
        reference_path=None,
        show=False,
        which="mean_and_std")



    import pdb; pdb.set_trace()






    if False:  # args.baseline_path:
        print("Parsing baseline folder")
        # deepobs.tensorflow.config.set_baseline_dir(args.baseline_path)
        deepobs.config.set_baseline_dir(args.baseline_path)
        # baseline_parser = deepobs.analyzer.analyze_utils.Analyzer(
        #     deepobs.tensorflow.config.get_baseline_dir()
        # )
        baseline_parser = deepobs.analyzer.shared_utils.SettingAnalyzer(
            deepobs.config.get_baseline_dir()
        )
    else:
        baseline_parser = None

    # Parse path folder
    print("Parsing results folder")
    ### folder_parser = deepobs.analyzer.analyze_utils.Analyzer(args.path)
    folder_parser = deepobs.analyzer.shared_utils.SettingAnalyzer(args.path)

    if args.get_best_run or args.full:
        deepobs.analyzer.analyze.get_best_run(folder_parser)
    if args.plot_lr_sensitivity or args.full:
        deepobs.analyzer.analyze.plot_lr_sensitivity(folder_parser,
                                                     baseline_parser)
    if args.plot_performance or args.full:
        deepobs.analyzer.analyze.plot_performance(folder_parser,
                                                  baseline_parser)
    if args.plot_table or args.full:
        deepobs.analyzer.analyze.plot_table(folder_parser, baseline_parser)


if __name__ == "__main__":
    main(**vars(read_args()))
