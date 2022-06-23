# -*- coding: utf-8 -*-

from . import abstract_runner, analyzer, config, scripts, tuner
from . import tensorflow, pytorch
from .version import __version__



# ##############################################################################
# # ANDRES PATCH: PATH MANAGER
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


# ##############################################################################
# # ANDRES PATCH: LR SCHEDULERS
# ##############################################################################
def long_trapezoidal_schedule(max_epochs, warmup_epochs=None,
                              decrease_from=None):
    if not warmup_epochs:
        warmup_epochs = round(0.1 * max_epochs)
    if not decrease_from:
        decrease_from = max_epochs - round(0.1 * max_epochs)
    lr_epochs = []
    lr_factors = []
    for warmup_step in range(1, warmup_epochs+1):
        factor = warmup_step / warmup_epochs
        lr_factors.append(factor)
        lr_epochs.append(warmup_step)
    for consistent_step in range(warmup_epochs+1, decrease_from + 1):
        lr_factors.append(lr_factors[-1])
        lr_epochs.append(consistent_step)
    step = 0
    # the +1 is to avoid last epoch with 0 learning rate
    decrease_duration = max_epochs - decrease_from + 1
    for decrease_step in range(decrease_from + 1, max_epochs+1):
        step += 1
        factor = 1 - step / decrease_duration
        lr_factors.append(factor)
        lr_epochs.append(decrease_step)
    return lr_epochs, lr_factors


def cosine_decay_schedule(max_epochs, min_lr = 0):
    lr_epochs = [i for i in range(1, max_epochs + 1)]
    lr_factors = []
    for step in range(1, max_epochs + 1):
        cosine_decay = 0.5  * (1 + math.cos(math.pi * (step / max_epochs)))
        decayed = (1-min_lr) * cosine_decay + min_lr
        lr_factors.append(decayed)
    return lr_epochs, lr_factors


def exponential_decay_schedule(decay_rate, max_epochs):
    lr_epochs = [i for i in range(1, max_epochs + 1)]
    lr_factors = []
    for step in range(1, max_epochs + 1):
        factor = decay_rate ** (step / max_epochs)
        lr_factors.append(factor)
    return lr_epochs, lr_factors


def cosine_decay_restarts(
        steps_for_cycle, max_epochs, increase_restart_interval_factor = 2,
        min_lr = 0, restart_discount = 0):
    lr_epochs = [i for i in range(1, max_epochs + 1)]
    lr_factors = []
    step = 0
    cycle = 0
    for epoch in range(1, max_epochs + 1):
        step += 1
        completed_fraction = step / steps_for_cycle
        cosine_decayed = 0.5  * (1 + math.cos(math.pi * completed_fraction))
        decayed = (1 - min_lr) * cosine_decayed + min_lr
        if decayed != min_lr:
            lr_factors.append(cosine_decayed)
        else:
            # Maybe discount here
            lr_factors.append(1 - (restart_discount * cycle))
        if completed_fraction == 1:
            step = 0
            cycle += 1
            # Increases the interval between restarts every restart
            steps_for_cycle = steps_for_cycle * increase_restart_interval_factor
    return lr_epochs, lr_factors


def warmup_decay_exponential(max_epochs, warmup_steps, warmup_iter_increase,
                             exp_decay_rate):
    lr_epochs = [i for i in range(1, max_epochs + 1)]
    lr_factors = []
    for warmup_step in range(1, warmup_steps+1):
        factor = 1 + warmup_step * warmup_iter_increase
        lr_factors.append(factor)
    max_epochs_exp = max_epochs - warmup_steps
    for decay_step in range(1, max_epochs_exp + 1):
        factor_decay = exp_decay_rate ** (decay_step / max_epochs_exp)
        decayed_fac = lr_factors[-1] * factor_decay
        lr_factors.append(decayed_fac)
    return lr_epochs, lr_factors
