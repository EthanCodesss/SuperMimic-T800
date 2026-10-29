"""Probe actual GPU setpoints, PD forces, saturation and replay action semantics."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=("g1", "t800"), required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--action-scale-mode", choices=("rl_lab", "torque_kp"), required=True)
    args = p.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    from scripts.t800_action_scaling import install, assert_runtime_scale
    action_spec = install(args.action_scale_mode)
    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.spec import NJ, NQ, joint_names
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(args.motion)], num_envs=2, replay_capacity=64,
        action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", anchor_obs=False,
    ), "cuda:0")
    try:
        assert_runtime_scale(env, action_spec)
        env.reset()
        torch.manual_seed(31)
        robot = env.scene["robot"]
        act = env.action_manager.get_term("joint_pos")
        io = StateIO(env)
        model = env.sim.mj_model
        names = joint_names()
        aids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint("robot/"+n).id)[0]) for n in names]
        aid = torch.tensor(aids, device=env.device)
        # Independently recover the selected robot's parameter table by names.
        if args.robot == "t800":
            from supermimic.robots.t800 import ACTION_SCALE, DEFAULT_POS
            scales = torch.tensor([action_spec["scale"][n] for n in names], device=env.device)
            offsets = torch.tensor([DEFAULT_POS[n] for n in names], device=env.device)
        else:
            import re
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, get_g1_robot_cfg
            init = get_g1_robot_cfg().init_state.joint_pos
            scales = torch.tensor([next(v for k,v in G1_ACTION_SCALE.items() if re.fullmatch(k,n)) for n in names],device=env.device)
            offsets = torch.tensor([next((v for k,v in init.items() if re.fullmatch(k,n)),0.) for n in names],device=env.device)
        checks = []
        for noise, magnitude in ((0., .7), (.1, .7), (0., 1000.)):
            env.reset()
            state = io.read()
            state[:, NQ+6:] = torch.linspace(-.4, .4, NJ, device=env.device)
            io.write(torch.arange(2, device=env.device), state)
            act.noise_std = noise
            raw = torch.linspace(-magnitude, magnitude, NJ, device=env.device).repeat(2,1)
            env.action_manager.process_action(raw)
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.forward()
            target = offsets + scales * act.executed_raw
            actual = env.sim.data.ctrl[:][:, aid]
            torch.testing.assert_close(actual, target, atol=2e-5, rtol=1e-6)
            torch.testing.assert_close(env.action_manager.action, raw)
            torch.testing.assert_close(robot.data.joint_pos_target, target, atol=2e-5, rtol=1e-6)
            gain = torch.tensor(model.actuator_gainprm[aids,0],device=env.device)
            kp = -torch.tensor(model.actuator_biasprm[aids,1],device=env.device)
            kd = -torch.tensor(model.actuator_biasprm[aids,2],device=env.device)
            expected = gain * actual - kp * state[:,7:NQ] - kd * state[:,NQ+6:]
            limits = torch.tensor(model.actuator_forcerange[aids],device=env.device)
            expected = expected.clamp(limits[:,0], limits[:,1])
            force = env.sim.data.actuator_force[:][:,aid]
            torch.testing.assert_close(force.double(), expected, atol=2e-3, rtol=1e-5)
            checks.append({"noise_std":noise,"action_magnitude":magnitude,
                           "max_target_error_rad":float((actual-target).abs().max()),
                           "max_force_error_nm":float((force-expected).abs().max()),
                           "saturated_actuators":int((force.abs() >= .99*limits[:,1]).sum())})
        # Verify the records consumed by the world model retain executed noise,
        # while the policy observations/action-rate reward keep the clean input.
        env.reset()
        act.noise_std = .1
        rec = env.recorder_manager.get_term("replay")
        clean = torch.linspace(-.1,.1,NJ,device=env.device).repeat(2,1)
        row = rec.buffer.ptr.clone()
        env.step(clean)
        assert not bool(env.reset_buf.any())
        ids = torch.arange(2, device=env.device)
        torch.testing.assert_close(rec.buffer.a[ids,row], clean)
        torch.testing.assert_close(rec.buffer.a_exec[ids,row], act.executed_raw)
        assert not torch.equal(clean, act.executed_raw)
        result={"robot":args.robot,"passed":True,"joint_names":list(names), "action_parameterization": action_spec,
                "actuator_indices_in_joint_order":aids,"control_dt_s":env.step_dt,
                "physics_dt_s":env.physics_dt,"probes":checks,
                "clean_executed_action_replay_separation":True,
                "scope":"Tests commanded positions and instantaneous PD forces; does not establish balance or dynamics-model gradient accuracy."}
        args.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
