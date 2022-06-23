# -*- coding: utf-8 -*-

from . import abstract_runner, analyzer, config, scripts, tuner
from . import tensorflow, pytorch
from .version import __version__



# ##############################################################################
# # ANDRES PATCHES
# ##############################################################################
import os
import pandas as pd


class CrowdedValleyPaths:
    """
    Path manager for the full benchmark results provided in the repository:
    SirRob1997/Crowded-Valley---Results.git

    * Results contain budgets
    * Budgets contain LR schedules
    * LR schedules contain tasks
    * Tasks contain optimizers
    * Optimizers contain settings
    * Settings contain runs with different random seeds: 1 JSON file per run

    This class provides functionality to arbitrarily filter the results.
    Usage example::

      pather = CrowdedValleyPaths(os.path.join(
          os.path.expanduser("~"), "git-work",
          "Crowded-Valley---Results", "results_main"))
      abspaths = pather(budgets={"large_budget"}, optimizers={"AdamOptimizer"},
                        tasks={"cifar100_allcnnc", "cifar10_3c3d"},
                        abspaths=True)
    """
    BUDGETS = {"large_budget", "medium_budget", "small_budget", "oneshot"}
    LR_SCHEDULES = {"none", "ltr", "cosine", "cosine_wr"}
    TASKS = {"cifar100_allcnnc", "cifar10_3c3d", "fmnist_2c2d", "fmnist_vae",
             "mnist_vae", "quadratic_deep", "svhn_wrn164", "tolstoi_char_rnn"}
    OPTIMIZERS = {"AdaBeliefOptimizer", "AMSBoundOptimizer",
                  "MomentumOptimizer", "AdaBoundOptimizer",
                  "AMSGrad", "NadamOptimizer",
                  "AdadeltaOptimizer", "GradientDescentOptimizer",
                  "NAGOptimizer", "AdagradOptimizer",
                  "LookaheadOptimizerMBGDMomentum", "RAdamOptimizer",
                  "AdamOptimizer", "LookaheadOptimizerRAdam",
                  "RMSPropOptimizer"}

    def __init__(self, results_root_path):
        """
        """
        self.df = self._create_table(results_root_path)

    @classmethod
    def _create_table(cls, results_root_path):
        """
        """
        paths = []
        for budget in cls.BUDGETS:
            for lrsched in cls.LR_SCHEDULES:
                for task in cls.TASKS:
                    for opt in cls.OPTIMIZERS:
                        abspath = os.path.join(
                            results_root_path, budget, lrsched, task, opt)
                        paths.append((budget, lrsched, task, opt, abspath))
        result = pd.DataFrame(
            paths, columns=("budget", "lrsched", "task", "opt", "abspath"))
        return result

    def __call__(self, budgets=None, lrscheds=None,
                 tasks=None, optimizers=None, abspaths=False):
        """
        :param bool abspaths: If true, only a list with the absolute paths is
          returned (as opposed to the full dataframe)
        """
        df = self.df
        # filter df
        if budgets is not None:
            assert all(x in self.BUDGETS for x in budgets), \
                f"Unknown budget in {budgets}"
            df = df.loc[df["budget"].isin(budgets)]
        if lrscheds is not None:
            assert all(x in self.LR_SCHEDULES for x in lrscheds), \
                f"Unknown lrsched in {lrscheds}"
            df = df.loc[df["lrsched"].isin(lrscheds)]
        if tasks is not None:
            assert all(x in self.TASKS for x in tasks), \
                f"Unknown task in {tasks}"
            df = df.loc[df["task"].isin(tasks)]
        if optimizers is not None:
            assert all(x in self.OPTIMIZERS for x in optimizers), \
                f"Unknown optimizer in {optimizers}"
            df = df.loc[df["opt"].isin(optimizers)]
        #
        if abspaths:
            return df["abspath"].tolist()
        else:
            return df
