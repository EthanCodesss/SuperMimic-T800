"""Activation rematerialization with unchanged full-batch upstream updates.

Retains PolicyTrainer.update verbatim, including noise draw order, critic
minibatches and optimizer steps. The policy and imagined step discard saved
activations and recompute them during backward; no trajectory partitioning.
"""
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from contextlib import nullcontext

from supermimic.imagination.policy_trainer import PolicyTrainer


class RecomputePolicyTrainer(PolicyTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        forward = self.policy.forward

        def recomputed_policy(obs_hist, goal, interface="motion"):
            if not torch.is_grad_enabled():
                return forward(obs_hist, goal, interface)
            return checkpoint(forward, obs_hist, goal, interface, use_reentrant=False,
                              preserve_rng_state=True)

        self.policy.forward = recomputed_policy
        step = self.step_fn

        def recomputed_step(s, a, a_prev, xi, ref):
            # Eager mjlab quaternion helpers are TorchScript functions, which
            # wrap checkpoint's internal early-stop exception in RuntimeError.
            # Recompute the full step in eager mode. Compiled training keeps
            # the validated default early-stop behavior.
            context = nullcontext() if self.cfg.compile else set_checkpoint_early_stop(False)
            with context:
                return checkpoint(step, s, a, a_prev, xi, ref, use_reentrant=False,
                                  preserve_rng_state=True)

        self.step_fn = recomputed_step
