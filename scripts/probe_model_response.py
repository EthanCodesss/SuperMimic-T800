"""Compare learned action sensitivities with central differences in real GPU physics.

Small local probes, not a global proof of model or contact-gradient correctness.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=("g1", "t800"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.spec import NJ, load_robot_model, validate_checkpoint_robot
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    validate_checkpoint_robot(ck["config"])
    windows = torch.load(args.windows, map_location="cuda", weights_only=True)
    assert windows["robot"] == args.robot
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(args.motion)], num_envs=1+2*NJ, record_transitions=False,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start",
    ), "cuda:0")
    try:
        env.reset()
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_robot_model(), device="cuda"),
            TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
            robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        wc = dict(ck["config"]["world_model"])
        wc["state_spec"] = StateSpecCfg(**wc["state_spec"])
        wm = WorldModel(WorldModelCfg(**wc), heads, "cuda")
        wm.load_state_dict(ck["wm"])
        wm.eval()
        io = StateIO(env)
        ids = torch.arange(env.num_envs, device="cuda")
        eye = torch.eye(NJ, device="cuda")
        samples=[]
        with torch.no_grad():
            for index in (0, 31, 63, 95):
                state = windows["states"][0,index].repeat(env.num_envs,1)
                base = windows["actions"][0,index].repeat(env.num_envs,1)
                xi = windows["xi"][index].repeat(env.num_envs,1)
                for eps in (.01, .05):
                    action = base.clone()
                    action[1:1+NJ] += eps*eye
                    action[1+NJ:] -= eps*eye
                    env.sim.reset()
                    io.write(ids, state)
                    env.action_manager.process_action(action)
                    for _ in range(env.cfg.decimation):
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step()
                        env.scene.update(dt=env.physics_dt)
                    actual = io.read()
                    prediction = wm.step(state,action,xi)
                    real_j = wm.spec.error_vector(actual[1:1+NJ],actual[1+NJ:])/(2*eps)
                    pred_j = wm.spec.error_vector(prediction[1:1+NJ],prediction[1+NJ:])/(2*eps)
                    def compare(a,b):
                        av,bv=a.flatten(),b.flatten()
                        return {"cosine":float(torch.nn.functional.cosine_similarity(av,bv,dim=0)),
                                "relative_l2_error":float(torch.linalg.vector_norm(av-bv)/torch.linalg.vector_norm(av).clamp_min(1e-8)),
                                "predicted_to_physics_norm_ratio":float(torch.linalg.vector_norm(bv)/torch.linalg.vector_norm(av).clamp_min(1e-8))}
                    samples.append({"window_index":index,"action_epsilon":eps,
                        "one_step_nse":float(wm.spec.normalized_sq_error(prediction[0],actual[0])),
                        "all_state":compare(real_j,pred_j),
                        "root_position_rotation":compare(real_j[:,:6],pred_j[:,:6]),
                        "joint_positions":compare(real_j[:,12:12+NJ],pred_j[:,12:12+NJ]),
                        "joint_velocities":compare(real_j[:,12+NJ:],pred_j[:,12+NJ:])})
        result={"robot":args.robot,"checkpoint":str(args.checkpoint),"windows":str(args.windows),
                "method":"Central differences of one 20 ms step, perturbing each normalized action independently; same start state and zero solver warmstart. Errors use StateSpec scales.",
                "caveat":"Four recorded states and two perturbation sizes; contact discontinuities can affect finite differences. Cosine measures agreement with a finite-difference response, not global model quality.",
                "samples":samples}
        args.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
