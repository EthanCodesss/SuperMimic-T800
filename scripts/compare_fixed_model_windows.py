"""Evaluate several same-robot checkpoints on identical recorded action sequences."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=("g1", "t800"), required=True)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    import torch
    from supermimic.robots.spec import validate_checkpoint_robot
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    torch.set_num_threads(4)
    w = torch.load(args.windows, map_location="cpu", weights_only=True)
    assert w["robot"] == args.robot
    results=[]
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_robot(ck["config"])
        cfg = dict(ck["config"]["world_model"])
        cfg["state_spec"] = StateSpecCfg(**cfg["state_spec"])
        # Dynamics-only evaluation does not call analytic FK/reward heads.
        wm = WorldModel(WorldModelCfg(**cfg), heads=None, device="cpu")
        wm.load_state_dict(ck["wm"])
        wm.eval()
        with torch.no_grad():
            error = wm.rollout_error(w["states"], w["actions"], w["xi"]).tolist()
        results.append({"checkpoint":str(path), "iteration":ck["iteration"],
                        "nse_by_horizon":error,
                        "selected_nse":{str(k):error[k-1] for k in (1,8,16,32) if k<=len(error)}})
    result={"robot":args.robot,"windows":str(args.windows),"backend":"CPU float32",
            "scope":"Fixed initial-policy physical windows; open loop recorded actions, not policy feedback rollouts or success rates.",
            "results":results}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
