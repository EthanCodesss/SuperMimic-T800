"""Trade activation memory for speed without changing the upstream update."""
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from supermimic.imagination.policy_trainer import PolicyTrainer


class SelectiveRecomputePolicyTrainer(PolicyTrainer):
    def __init__(self, *args, recompute_mode, **kwargs):
        super().__init__(*args, **kwargs)
        if recompute_mode == "policy":
            forward = self.policy.forward

            def recomputed_policy(obs_hist, goal, interface="motion"):
                if not torch.is_grad_enabled():
                    return forward(obs_hist, goal, interface)
                return checkpoint(forward, obs_hist, goal, interface, use_reentrant=False,
                                  preserve_rng_state=True)

            self.policy.forward = recomputed_policy
        elif recompute_mode == "step":
            step = self.step_fn

            def recomputed_step(s, a, a_prev, xi, ref):
                context = nullcontext() if self.cfg.compile else set_checkpoint_early_stop(False)
                with context:
                    return checkpoint(step, s, a, a_prev, xi, ref, use_reentrant=False,
                                      preserve_rng_state=True)

            self.step_fn = recomputed_step
        else:
            raise ValueError(f"Expected policy or step recomputation, got {recompute_mode}")
