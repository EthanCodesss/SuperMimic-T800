"""Compare actual upstream updates with the opt-in memory implementation.

Uses real motion/FK/reward/GAE and nonzero dynamics action sensitivity. Small
models serve CPU float64 tests; --full-model uses original hidden sizes and
the public trained G1 checkpoint. No GPU simulator is needed for this test.
"""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_strict import select_source, verify_sources, sha


def setup(robot, device, dtype, full_model, anchor):
    import torch
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
    if robot == "g1":
        from supermimic.data.motion_convert import load_g1_model
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
        model = load_g1_model()
        rc = get_g1_robot_cfg()
        names = [model.joint(i).name for i in range(1, model.njnt)]
        initial = [next((v for k, v in rc.init_state.joint_pos.items() if re.fullmatch(k, n)), 0.) for n in names]
        motion = ROOT / "data/npz/lafan1/walk1_subject1.npz"
    else:
        from supermimic.robots.t800 import load_t800_model, DEFAULT_POS, JOINT_NAMES
        model = load_t800_model()
        initial = [DEFAULT_POS[n] for n in JOINT_NAMES]
        motion = ROOT / "data/npz/t800_single/walk2_steady.npz"
    q0 = torch.tensor(initial, dtype=dtype, device=device)
    ranges = torch.tensor(model.jnt_range[1:].copy(), dtype=dtype, device=device)
    middle, half = ranges.mean(-1), (ranges[:, 1] - ranges[:, 0]) * .45
    limits = torch.stack([middle - half, middle + half], -1)
    heads = AnalyticHeads(G1Kinematics(model, device=device, dtype=dtype), TRACKED_BODIES,
                          ANCHOR_BODY, ROOT_BODY, limits, q0)
    wc = WorldModelCfg() if full_model else WorldModelCfg(hidden=(64, 48), term_hidden=(16, 16))
    pc = PolicyCfg() if full_model else PolicyCfg(encoder_hidden=(64, 48), decoder_hidden=(64, 48))
    pc.goal_dim += 9 if anchor else 0
    wm = WorldModel(wc, heads, device).to(dtype=dtype)
    wm.spec = StateSpec(wc.state_spec, device, dtype)
    policy = Policy(pc).to(device=device, dtype=dtype)
    critic = Critic(CriticCfg(), pcfg=pc).to(device=device, dtype=dtype)
    if full_model and robot == "g1":
        assert not anchor, "Public checkpoint has no anchor observations"
        with torch.serialization.safe_globals([Path, type(Path("."))]):
            ck = torch.load(ROOT / ".cache/downloads/e8_boneseed_ckpt_0100000.pt", map_location=device, weights_only=True)
        wm.load_state_dict(ck["wm"])
        policy.load_state_dict(ck["policy"])
        critic.load_state_dict(ck["critic"])
        del ck
    else:
        # Zero-initialized dynamics would make action-gradient parity too weak.
        with torch.no_grad():
            last = [m for m in wm.dynamics.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.weight.normal_(std=.002)
    lib = MotionLibrary([motion], TRACKED_BODIES, q0, device=device, dtype=dtype)
    return wm, policy, critic, lib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("g1", "t800"), default="g1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--full-model", action="store_true")
    p.add_argument("--checkpoint", type=Path,
                   help="Compare disposable copies of trained weights and optimizer states; never overwrite the checkpoint")
    p.add_argument("--implementation", choices=("original", "microbatch", "recompute"), default="microbatch")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--match-env-backends", action="store_true",
                   help="Use mjlab's training-time TF32/backend settings for both compared implementations")
    p.add_argument("--recompute-mode", choices=("all", "step", "policy"), default="all")
    p.add_argument("--anchor", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--collect-all", action="store_true", help="Diagnostic: record all tolerance failures before exiting nonzero")
    p.add_argument("--batch", type=int, default=11)
    p.add_argument("--actor-chunk", type=int, default=4)
    p.add_argument("--critic-chunk", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    verify_sources()
    source = select_source(args.robot)
    import torch
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.kinematics.rotations import quat_apply_inverse
    import supermimic.imagination.policy_trainer as original
    from optimizations.policy_trainer import MicrobatchPolicyTrainer
    from optimizations.recompute import RecomputePolicyTrainer
    torch.set_num_threads(4)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=args.match_env_backends)
    torch.manual_seed(131)
    dtype = getattr(torch, args.dtype)
    wm, policy, critic, lib = setup(args.robot, args.device, dtype, args.full_model, args.anchor)
    trained = None
    if args.checkpoint:
        assert args.full_model and dtype == torch.float32, "Trained-checkpoint audit requires the full float32 model"
        with torch.serialization.safe_globals([Path, type(Path("."))]):
            trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        assert trained["config"].get("robot", "g1") == args.robot
        assert trained["config"]["env"].get("anchor_obs", False) == args.anchor
        wm.load_state_dict(trained["wm"])
        policy.load_state_dict(trained["policy"])
        critic.load_state_dict(trained["critic"])
    cfg = original.PolicyTrainerCfg(batch_size=args.batch, compile="default" if args.compile else None,
                                    anchor_obs=args.anchor, obs_noise=not args.no_noise)
    if trained:
        expected = asdict(cfg)
        actual = trained["config"]["policy_trainer"]
        for key in expected:
            if key != "batch_size":
                assert json.dumps(expected[key]) == json.dumps(actual[key]), (key, expected[key], actual[key])
    ref = original.PolicyTrainer(wm, policy, critic, lib, copy.deepcopy(cfg))
    models = (copy.deepcopy(wm), copy.deepcopy(policy), copy.deepcopy(critic), lib, copy.deepcopy(cfg))
    if args.implementation == "original":
        micro = original.PolicyTrainer(*models)
    elif args.implementation == "recompute" and args.recompute_mode != "all":
        from optimizations.selective_recompute import SelectiveRecomputePolicyTrainer
        micro = SelectiveRecomputePolicyTrainer(*models, recompute_mode=args.recompute_mode)
    elif args.implementation == "recompute":
        micro = RecomputePolicyTrainer(*models)
    else:
        micro = MicrobatchPolicyTrainer(*models, actor_chunk_size=args.actor_chunk,
                                       critic_chunk_size=args.critic_chunk)
    if trained:
        for trainer in (ref, micro):
            trainer.opt.load_state_dict(copy.deepcopy(trained["policy_opt"]))
            trainer.critic_opt.load_state_dict(copy.deepcopy(trained["critic_opt"]))
        del trained
    frames = torch.linspace(15, int(lib.clip_last[0]) - 1, args.batch, device=args.device).long()
    def state_at(fr):
        r = lib.ref(fr)
        return torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w,
                          quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w), r.joint_vel], -1)
    hist_frames = frames[None] + torch.arange(-9, 1, device=args.device)[:, None]
    hist_states = state_at(hist_frames.reshape(-1)).reshape(10, args.batch, -1)
    nj = lib.default_joint_pos.numel()
    xi = torch.zeros(args.batch, nj + 4, device=args.device, dtype=dtype)
    xi[:, 0] = 1.
    xi[:, 1:1 + nj] = torch.linspace(-.002, .002, nj, device=args.device, dtype=dtype)
    actions = .02 * torch.randn(10, args.batch, nj, device=args.device, dtype=dtype)
    start = StartBatch(hist_states[-1], xi, frames, actions[-1], hist_states, actions)
    class FixedBuffer:
        def sample_starts(self, n, history):
            assert (n, history) == (args.batch, 10)
            return start
    buffer = FixedBuffer()
    errors = {}
    failures = []
    atol, rtol = (2e-10, 2e-8) if dtype == torch.float64 else (5e-6, 5e-4)
    def compare(name, a, b, parameter_before=None):
        a, b = a.detach().cpu(), b.detach().cpu()
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        delta = (a.double() - b.double()).abs()
        errors[name] = {"max_abs": float(delta.max()) if a.numel() else 0.,
                        "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-30))}
        # Upstream term_sum is float32 even during a float64 rollout. Chunked
        # logging reductions therefore have float32 rounding; gradients do not.
        diag = "/metric/imag_r/" in name
        try:
            if dtype == torch.float32 and "/gradient" in name:
                # Near-zero coordinates make elementwise relative tests unstable.
                # Bound the full update direction error, retaining max-absolute
                # errors in the report; float64 tests check elementwise algebra.
                assert errors[name]["relative_l2"] <= 1e-3, (name, errors[name])
                if args.implementation not in ("recompute", "original"):
                    return
            if dtype == torch.float32 and parameter_before is not None:
                magnitude = (a.double() - parameter_before.double().cpu()).norm()
                rel = float(delta.norm() / magnitude.clamp_min(1e-30))
                errors[name]["relative_l2_to_reference_parameter_update"] = rel
                assert rel <= 1e-3, (name, errors[name])
                if args.implementation not in ("recompute", "original"):
                    return
            torch.testing.assert_close(a, b, atol=max(atol, 2e-7) if diag else atol,
                                       rtol=max(rtol, 2e-6) if diag else rtol,
                                       msg=lambda msg: f"{name}: {msg}")
        except AssertionError as exc:
            failures.append({"name": name, "detail": str(exc)})
            if not args.collect_all:
                raise
    def flat_grad(module):
        return torch.cat([p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]).clone()
    baseline_wm = {k: v.clone() for k, v in wm.state_dict().items()}
    summaries = []
    try:
        for update in range(2):
            before = {kind: {k: v.detach().cpu().clone() for k, v in getattr(ref, kind).state_dict().items()}
                      for kind in ("policy", "critic")}
            ref_grads, ref_targets, micro_grads, micro_targets = [], [], [], []
            real_gae = original.gae_targets
            def capture_gae(*a, **kw):
                result = real_gae(*a, **kw)
                ref_targets.append(result.detach().clone())
                return result
            def capture_grad(module):
                ref_grads.append(flat_grad(module))
            def observer(event, **data):
                if event in ("actor_grad", "critic_grad"):
                    micro_grads.append(flat_grad(data["module"]))
                if event == "trajectory":
                    micro_targets.append(data["targets"].clone())
            micro.audit_observer = observer
            seed = 501 + update
            torch.manual_seed(seed)
            with patch.object(original, "gae_targets", capture_gae), patch.object(original, "allreduce_grads", capture_grad):
                a = ref.update(buffer)
            rng_ref = torch.cuda.get_rng_state() if args.device.startswith("cuda") else torch.get_rng_state()
            torch.manual_seed(seed)
            if args.implementation in ("recompute", "original"):
                def micro_gae(*a, **kw):
                    result = real_gae(*a, **kw)
                    micro_targets.append(result.detach().clone())
                    return result
                def micro_grad(module):
                    micro_grads.append(flat_grad(module))
                with patch.object(original, "gae_targets", micro_gae), patch.object(original, "allreduce_grads", micro_grad):
                    b = micro.update(buffer)
            else:
                b = micro.update(buffer)
            rng_micro = torch.cuda.get_rng_state() if args.device.startswith("cuda") else torch.get_rng_state()
            assert torch.equal(rng_ref, rng_micro), "Random-number consumption changed"
            assert len(ref_grads) == len(micro_grads) == 21, "Must retain one actor and twenty critic updates"
            compare(f"update{update}/gae", ref_targets[0], torch.cat(micro_targets, 1))
            for i, (x, y) in enumerate(zip(ref_grads, micro_grads)):
                compare(f"update{update}/gradient{i}", x, y)
            for name in a:
                compare(f"update{update}/metric/{name}", torch.tensor(a[name], dtype=dtype), torch.tensor(b[name], dtype=dtype))
            for kind in ("policy", "critic", "wm"):
                x, y = getattr(ref, kind), getattr(micro, kind)
                for key in x.state_dict():
                    compare(f"update{update}/{kind}/{key}", x.state_dict()[key], y.state_dict()[key],
                            parameter_before=before[kind][key] if kind in before else None)
            for kind in ("opt", "critic_opt"):
                x, y = getattr(ref, kind).state_dict(), getattr(micro, kind).state_dict()
                assert x["param_groups"] == y["param_groups"]
                for index in x["state"]:
                    for key in x["state"][index]:
                        compare(f"update{update}/{kind}/{index}/{key}", x["state"][index][key], y["state"][index][key])
            for key, value in baseline_wm.items():
                assert torch.equal(value, micro.wm.state_dict()[key]), "Frozen world model changed"
            summaries.append({"update": update + 1, "rng_equal": True, "actor_steps": update + 1,
                              "critic_steps": 20 * (update + 1), "actor_gradient_norm": a["policy_grad_norm"]})
            print(f"[PARITY] update {update + 1} compared; tolerance failures so far: {len(failures)}", flush=True)
        passed = not failures
    except Exception:
        passed = False
        raise
    finally:
        result = {"passed": passed, "source": str(source), "arguments": vars(args), "recipe": asdict(cfg),
                  "checkpoint_sha256": sha(args.checkpoint) if args.checkpoint else None,
                  "optimizer_initialization": "trained checkpoint copies" if args.checkpoint else "fresh Adam states",
                  "model_copy_strategy": "independent module copies, shared immutable motion library",
                  "matmul_precision": torch.backends.cuda.matmul.fp32_precision,
                  "cudnn_precision": torch.backends.cudnn.conv.fp32_precision,
                  "torch": str(torch.__version__), "validator_sha256": sha(Path(__file__)),
                  "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "optimizations").glob("*.py"))},
                  "tolerance": {"atol": atol, "rtol": rtol,
                                "float32_gradients": "relative L2 <= 1e-3; max absolute error also reported",
                                "float32_parameters": "difference L2 / reference parameter update L2 <= 1e-3, per tensor",
                                "recompute_requires_original_elementwise_tolerance_too": True,
                                "reward_term_logging": "Upstream float32 accumulator: atol >= 2e-7, rtol >= 2e-6"},
                  "updates": summaries, "comparisons": errors, "failures": failures,
                  "scope": "Two optimizer updates, real FK/reward, full 32-step BPTT, 10 history frames, fixed matching noise and RNG, all 20 critic steps per update. Chunk sizes apply only to microbatch mode. Not convergence equivalence."}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"passed": passed, "comparisons": len(errors),
                      "max_abs": max(v["max_abs"] for v in errors.values())}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
