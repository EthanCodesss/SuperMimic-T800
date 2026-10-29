"""Full effective batches, trajectory microbatches, and unchanged Adam cadence.

Only independent trajectories are partitioned. Each retains the full BPTT
horizon. Actor gradients are clipped once, after weighted accumulation. Critic
targets keep upstream time-major order and each original minibatch retains one
Adam step. No mixed precision, checkpointing or truncated state gradients.
"""
from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import OBS_NOISE_HALF
from supermimic.training.distributed import allreduce_grads

from optimizations.rollout import imagine


@dataclass
class RolloutNoise:
    observation: list[torch.Tensor]
    anchor: list[torch.Tensor]

    def slice(self, lo, hi):
        return RolloutNoise([x[lo:hi] for x in self.observation], [x[lo:hi] for x in self.anchor])

    @staticmethod
    @torch.no_grad()
    def sample(start, cfg):
        """Same rand_like shapes, layout and ordering as one upstream rollout."""
        if not cfg.obs_noise:
            return None
        n = start.state.shape[0]
        obs_template = start.state.new_empty(n, len(OBS_NOISE_HALF))
        anchor_template = start.state.new_empty(n, 9)
        obs = [torch.rand_like(obs_template) for _ in range(start.hist_states.shape[0])]
        anchor = [torch.rand_like(anchor_template)] if cfg.anchor_obs else []
        for _ in range(cfg.horizon):
            obs.append(torch.rand_like(obs_template))
            if cfg.anchor_obs:
                anchor.append(torch.rand_like(anchor_template))
        return RolloutNoise(obs, anchor)


def slice_start(start, lo, hi):
    return StartBatch(start.state[lo:hi], start.xi[lo:hi], start.frame[lo:hi],
                      start.prev_action[lo:hi], start.hist_states[:, lo:hi], start.hist_actions[:, lo:hi])


class MicrobatchPolicyTrainer(PolicyTrainer):
    def __init__(self, *args, actor_chunk_size=512, critic_chunk_size=4096, **kwargs):
        super().__init__(*args, **kwargs)
        if actor_chunk_size < 1 or critic_chunk_size < 1:
            raise ValueError("Chunk sizes must be positive")
        self.actor_chunk_size = actor_chunk_size
        self.critic_chunk_size = critic_chunk_size
        # Optional validation observer receives detached data; production keeps none.
        self.audit_observer = None

    def observe(self, event, **data):
        if self.audit_observer is not None:
            self.audit_observer(event, **data)

    def update(self, buffer, xi_norm_fn=None):
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        n = start.state.shape[0]
        noise = RolloutNoise.sample(start, cfg)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        metrics = {}
        features = goals = errors = targets = None
        value_sum = start.state.new_zeros(())
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        try:
            for lo in range(0, n, self.actor_chunk_size):
                hi = min(lo + self.actor_chunk_size, n)
                weight = (hi - lo) / n
                ro = imagine(self.wm, self.policy, self.lib, slice_start(start, lo, hi),
                             cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise,
                             step_fn=self.step_fn, anchor_obs=cfg.anchor_obs,
                             noise=None if noise is None else noise.slice(lo, hi))
                feat, goal, er = self._critic_inputs(ro)
                v_H = self.critic(feat[-1], goal[-1], xi_norm[lo:hi], er[-1]) if cfg.use_critic else None
                J = policy_objective(ro, v_H, cfg.gamma)
                (-J * weight / cfg.horizon).backward()
                with torch.no_grad():
                    local = {
                        "imag_objective": J.detach(),
                        "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean(),
                        "imag_reward_mean": ro.rewards.mean(),
                        "imag_survival_H": ro.survival[-1].mean(),
                        "imag_dhat_mean": ro.dhat[1:].mean(),
                        "action_abs_mean": ro.actions.abs().mean(),
                    }
                    for key, value in local.items():
                        metrics[key] = metrics.get(key, 0.) + value.detach() * weight
                    for key, value in ro.reward_terms.items():
                        key = f"imag_r/{key}"
                        metrics[key] = metrics.get(key, 0.) + value * weight
                    if cfg.use_critic:
                        H1, m, _ = feat.shape
                        xi_rep = xi_norm[None, lo:hi].expand(H1, m, -1)
                        values = self.critic(feat.detach().reshape(H1 * m, -1),
                                             goal.detach().reshape(H1 * m, -1),
                                             xi_rep.reshape(H1 * m, -1),
                                             er.detach().reshape(H1 * m, -1)).reshape(H1, m)
                        target = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values,
                                             cfg.gamma, cfg.gae_lambda)
                        if features is None:
                            features = feat.new_empty(cfg.horizon, n, feat.shape[-1])
                            goals = goal.new_empty(cfg.horizon, n, goal.shape[-1])
                            errors = er.new_empty(cfg.horizon, n, er.shape[-1])
                            targets = target.new_empty(cfg.horizon, n)
                        features[:, lo:hi].copy_(feat[:-1])
                        goals[:, lo:hi].copy_(goal[:-1])
                        errors[:, lo:hi].copy_(er[:-1])
                        targets[:, lo:hi].copy_(target)
                        value_sum += values.sum()
                        self.observe("trajectory", lo=lo, hi=hi, targets=target, values=values,
                                     states=ro.states.detach(), rewards=ro.rewards.detach(),
                                     actions=ro.actions.detach())
                        del values, target, xi_rep
                # Do not retain a previous chunk's autograd graph into the next one.
                del ro, feat, goal, er, v_H, J, local
            allreduce_grads(self.policy)
            self.observe("actor_grad", module=self.policy)
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.opt.step()
            metrics["policy_grad_norm"] = gn.detach()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        del noise

        if cfg.use_critic:
            X_feat = features.reshape(-1, features.shape[-1])
            X_goal = goals.reshape(-1, goals.shape[-1])
            X_er = errors.reshape(-1, errors.shape[-1])
            X_xi = xi_norm[None].expand(cfg.horizon, n, -1).reshape(-1, xi_norm.shape[-1])
            Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            if mb < 1:
                raise ValueError("Critic minibatch would be empty")
            for epoch in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb:(i + 1) * mb]
                    self.critic_opt.zero_grad(set_to_none=True)
                    last = Y.new_zeros(())
                    for lo in range(0, mb, self.critic_chunk_size):
                        part = idx[lo:lo + self.critic_chunk_size]
                        pred = self.critic(X_feat[part], X_goal[part], X_xi[part], X_er[part])
                        loss = torch.nn.functional.mse_loss(pred, Y[part])
                        weight = part.numel() / mb
                        (loss * weight).backward()
                        last += loss.detach() * weight
                        del pred, loss
                    allreduce_grads(self.critic)
                    self.observe("critic_grad", module=self.critic, epoch=epoch, minibatch=i,
                                 indices=idx, loss=last)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
            metrics["critic_loss"] = last
            metrics["critic_value_mean"] = value_sum / ((cfg.horizon + 1) * n)
            metrics["critic_target_mean"] = Y.mean()
        return {key: float(value) for key, value in metrics.items()}
