"""Run once per robot in a separate process: SUPERMIMIC_ROBOT=t800 pytest ..."""
import numpy as np
import torch

from supermimic.robots.spec import NJ, NQ, NV, STATE_DIM, DELTA_DIM, XI_DIM, GOAL_FEAT_DIM, OBS_FRAME_DIM, ROBOT, load_robot_model, validate_checkpoint_robot


def test_robot_state_policy_and_gradients():
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
    from supermimic.world_model.dynamics import DynamicsMLP
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    from supermimic.policy.nets import Policy, PolicyCfg
    model = load_robot_model()
    assert (model.nq, model.nv) == (NQ, NV)
    state = torch.zeros(3, STATE_DIM)
    state[:, :NQ] = torch.tensor(model.qpos0, dtype=torch.float32)
    xi = torch.zeros(3, XI_DIM)
    xi[:, 0] = 1.
    spec = StateSpec(StateSpecCfg(), "cpu")
    net = DynamicsMLP(hidden=(32, 32))
    delta = net(spec.canonical(state), torch.zeros(3, NJ), spec.normalize_xi(xi))
    assert delta.shape == (3, DELTA_DIM)
    next_state = spec.integrate(state, delta)
    obs = analytic_observation(next_state, torch.zeros(3, NJ), xi, torch.zeros(NJ))
    assert obs.shape == (3, OBS_FRAME_DIM)
    history = flatten_history_term_major(obs.repeat(10, 1, 1))
    policy = Policy(PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,)))
    action = policy(history, torch.randn(3, 10 * GOAL_FEAT_DIM))
    assert action.shape == (3, NJ)
    action.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in net.parameters())


def test_checkpoint_robot_guard():
    import pytest
    validate_checkpoint_robot({"robot": ROBOT})
    with pytest.raises(ValueError, match="Checkpoint robot"):
        validate_checkpoint_robot({"robot": "t800" if ROBOT == "g1" else "g1"})


def test_t800_actuators_match_named_parameters():
    import pytest
    if ROBOT != "t800":
        pytest.skip("T800-only actuator configuration")
    from mjlab.entity import Entity
    from supermimic.robots.t800 import get_t800_robot_cfg, JOINT_NAMES, ACTUATOR_GROUPS, DEFAULT_POS
    entity = Entity(get_t800_robot_cfg())
    action_indices, action_names = entity.find_joints_by_actuator_names((".*",))
    assert action_indices == list(range(25))
    assert tuple(action_names) == JOINT_NAMES
    model = entity.spec.compile()
    assert model.nu == 25
    for indices, kp, kd, limit, armature, velocity in ACTUATOR_GROUPS:
        for index in indices:
            joint = model.joint(JOINT_NAMES[index])
            actuator_ids = np.where(model.actuator_trnid[:, 0] == joint.id)[0]
            assert len(actuator_ids) == 1
            aid = actuator_ids[0]
            assert model.actuator_gainprm[aid, 0] == kp
            assert model.actuator_biasprm[aid, 2] == -kd
            np.testing.assert_allclose(model.actuator_forcerange[aid], [-limit, limit])
            assert model.dof_armature[joint.dofadr[0]] == armature
