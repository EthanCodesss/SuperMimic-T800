"""Read pinned EngineAI parameter expressions without importing Isaac Lab.

This checks named control parameters, not PhysX/MuJoCo dynamic equivalence.
Only literals, earlier constants, arithmetic and config keyword arguments are read.
"""
import ast
import hashlib
import json
import operator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_value(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [read_value(v, constants) for v in node.elts]
    if isinstance(node, ast.Dict):
        return {read_value(k, constants): read_value(v, constants) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -read_value(node.operand, constants)
    if isinstance(node, ast.BinOp):
        op = {ast.Mult: operator.mul, ast.Pow: operator.pow, ast.Add: operator.add}[type(node.op)]
        return op(read_value(node.left, constants), read_value(node.right, constants))
    if isinstance(node, ast.Call) and not node.args:
        return {k.arg: read_value(k.value, constants) for k in node.keywords}
    raise ValueError(f"Unsupported source expression: {ast.dump(node)}")


def main():
    from supermimic.robots.t800 import (
        JOINT_NAMES, DEFAULT_POS, ACTION_SCALE, ACTUATOR_GROUPS, TRACKED_BODIES,
        ANCHOR_BODY, get_t800_robot_cfg,
    )
    path = ROOT / "vendor/engineai_rl_lab_reference/t800.py"
    constants = {}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = read_value(statement.value, constants)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            constants[statement.target.id] = read_value(statement.value, constants)
    upstream = constants["T800_CYLINDER_CFG"]
    actual = {}
    for indices, kp, kd, torque, armature, velocity in ACTUATOR_GROUPS:
        for i in indices:
            actual[JOINT_NAMES[i]] = dict(stiffness=kp, damping=kd, effort_limit_sim=torque,
                                        armature=armature, velocity_limit_sim=velocity)
    fields = tuple(next(iter(actual.values())))
    rows = []
    for actuator in upstream["actuators"].values():
        for joint in actuator["joint_names_expr"]:
            for field in fields:
                assert abs(actual[joint][field] - actuator[field]) < 1e-12, (joint, field)
            assert DEFAULT_POS[joint] == upstream["init_state"]["joint_pos"][joint], joint
            assert ACTION_SCALE[joint] == constants["T800_ACTION_SCALE"][joint], joint
            rows.append({"joint": joint, **actual[joint], "default_position": DEFAULT_POS[joint],
                         "action_scale": ACTION_SCALE[joint],
                         "official_delay_physics_steps": [actuator["min_delay"], actuator["max_delay"]]})
    assert list(JOINT_NAMES) == upstream["joint_sdk_names"]
    local_cfg = get_t800_robot_cfg()
    assert list(local_cfg.init_state.pos) == upstream["init_state"]["pos"]
    assert local_cfg.articulation.soft_joint_pos_limit_factor == upstream["soft_joint_pos_limit_factor"]
    flat = ROOT / "vendor/engineai_rl_lab_reference/flat_env_cfg.py"
    assignments = {ast.unparse(s.targets[0]): s.value for s in ast.walk(ast.parse(flat.read_text()))
                   if isinstance(s, ast.Assign) and len(s.targets) == 1}
    assert read_value(assignments["self.commands.motion.body_names"], {}) == list(TRACKED_BODIES)
    assert read_value(assignments["self.commands.motion.anchor_body_name"], {}) == ANCHOR_BODY
    result = {
        "official_commit": "c53f7ec614e34f1ea514cefb935894082f3d4c2a",
        "official_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "matched": True, "joint_count": len(rows),
        "checks": ["SDK joint order", "default joint/root pose", "PD gains", "effort limits",
                   "armature", "action scale", "soft limit factor", "tracked bodies", "anchor body"],
        "remaining_differences": [
            "Official random delay of 1-3 physics steps is omitted in the adapter",
            "Velocity ratings match numerically but are not enforced as velocity_limit_sim",
            "GMR/SDK MJCF versus RL Lab USD mass/inertia/collision equivalence is unverified",
            "MuJoCo implicit PD/contact solver differs from PhysX; numeric parameter agreement is not dynamics parity",
            "SuperMimic rewards/observations/terminations are retained instead of copying the EngineAI PPO task",
        ],
        "joints": sorted(rows, key=lambda r: JOINT_NAMES.index(r["joint"])),
    }
    out = ROOT / "reports/upstream-audit/t800-source-parameters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("matched", "joint_count", "checks", "remaining_differences")}, indent=2))


if __name__ == "__main__":
    main()
