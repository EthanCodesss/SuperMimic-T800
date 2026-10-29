"""Local 3D inspection of T800 reference poses and recorded simulation trajectories.

This viewer sets qpos and computes kinematics; it never runs a policy or mj_step.
Run: conda run -n supermimic python scripts/view_motions.py
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from supermimic.robots.t800 import JOINT_NAMES, REPO_ROOT, ROOT_BODY, get_spec


CLIPS = (
    ("参考 · 训练使用的行走", "data/npz/t800_single/walk2_steady.npz",
     "LAFAN1 walk2_subject1，第 300–419 帧，经众擎 GMR 重定向。"
     "这是逐帧指定姿态的参考回放，尚未证明动力学可跟踪。"),
    ("参考 · 行走片段 1", "data/npz/t800_lafan1/walk1_short.npz",
     "LAFAN1 walk1_subject1，第 60–179 帧，经众擎 GMR 重定向。参考姿态回放。"),
    ("参考 · 行走片段 2", "data/npz/t800_lafan1/walk2_short.npz",
     "LAFAN1 walk2_subject1，第 60–179 帧，经众擎 GMR 重定向。参考姿态回放。"),
    ("参考 · 众擎舞蹈示例", "data/npz/t800_vendor/dance_t800.npz",
     "众擎项目 dance_t800.csv 转换的参考姿态。不是本项目训练得到的策略表现。"),
    ("参考 · 合成站立", "data/npz/t800_standing/standing.npz",
     "官方默认关节角生成的固定姿态，初始足底距地 2 毫米。不是动作捕捉数据。"),
    ("物理记录 · 训练后行走", "reports/t800-10000.trajectory.npz",
     "10,000 步试训后的确定性评估记录，约 1.04 秒因身体高度下降终止。"
     "这里回放保存的实际状态，不重新运行策略。"),
    ("物理记录 · 固定姿态 PD", "reports/t800-standing-pd.trajectory.npz",
     "固定关节目标的 PD 控制记录，约 1.38 秒因身体高度下降终止。"
     "该控制器没有主动平衡功能；这里回放实际状态。"),
)


def load_clip(path: Path, nq: int) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as motion:
        fps = float(motion["fps"].reshape(-1)[0])
        if "qpos" in motion:
            qpos = motion["qpos"].copy()
        else:
            if tuple(motion["joint_names"].tolist()) != JOINT_NAMES:
                raise ValueError(f"Joint order differs from the T800 model: {path}")
            root = motion["body_names"].tolist().index(ROOT_BODY)
            qpos = np.concatenate([
                motion["body_pos_w"][:, root], motion["body_quat_w"][:, root],
                motion["joint_pos"],
            ], axis=1)
    if qpos.ndim != 2 or qpos.shape[1] != nq or len(qpos) < 2:
        raise ValueError(f"Invalid qpos shape in {path}: {qpos.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid pose values or fps: {path}")
    return qpos, fps


def foot_bottom(model: mujoco.MjModel, data: mujoco.MjData, ids: list[int]) -> float:
    heights = []
    for i in ids:
        size = model.geom_size[i]
        rz = data.geom_xmat[i].reshape(3, 3)[2]
        kind = model.geom_type[i]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = np.abs(rz) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = abs(rz[2]) * size[1] + size[0]
        else:
            raise ValueError(f"Unsupported foot geometry: {kind}")
        heights.append(data.geom_xpos[i, 2] - radius)
    return float(min(heights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    clips = {name: (*load_clip(REPO_ROOT / path, model.nq), description)
             for name, path, description in CLIPS if (REPO_ROOT / path).exists()}
    if not clips:
        raise FileNotFoundError("No T800 reference or trajectory files found")
    feet = [[i for i in range(model.ngeom)
             if model.geom(i).name.startswith(f"LINK_FOOT_{side}_collision_")]
            for side in ("L", "R")]
    server = viser.ViserServer(host=args.host, port=args.port, label="T800 · 动作检查")
    server.gui.configure_theme(show_share_button=False, brand_color=(25, 128, 118))
    server.scene.set_up_direction("+z")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    server.scene.add_grid("/ground", width=40., height=40., cell_size=.2,
                          section_size=1., plane="xy", plane_opacity=.8)
    server.gui.add_markdown("## T800 动作检查\n拖动旋转视角，滚轮缩放；右键拖动平移。")
    select = server.gui.add_dropdown("动作", options=tuple(clips))
    description = server.gui.add_markdown("")
    playing = server.gui.add_checkbox("播放", initial_value=True)
    loop = server.gui.add_checkbox("循环播放", initial_value=True)
    speed = server.gui.add_dropdown("播放速度", options=("0.1×", "0.25×", "0.5×", "1×", "2×"), initial_value="1×")
    frame = server.gui.add_slider("帧", min=0, max=len(clips[select.value][0])-1, step=1, initial_value=0)
    previous = server.gui.add_button("上一帧")
    following = server.gui.add_button("下一帧")
    view = server.gui.add_dropdown("视角", options=("斜前方", "正面", "侧面"))
    reset_camera = server.gui.add_button("重置视角")
    collisions = server.gui.add_checkbox("显示碰撞几何", initial_value=False)
    status = server.gui.add_markdown("")
    server.gui.add_markdown("地面小格为 **20 厘米**。足底高度是几何计算，不代表接触力。"
                            "\n\n循环末尾会跳回首帧；片段没有做首尾衔接。所有页面控制由访问者共享。")
    lock = threading.RLock()
    state = {"frame": 0., "changed": True, "camera": True}

    def set_camera(client):
        q = clips[select.value][0]
        center = (q[:, :3].min(axis=0) + q[:, :3].max(axis=0)) / 2
        center[2] = .85
        distance = max(3.7, float(np.ptp(q[:, :2], axis=0).max()) * 1.3 + 2.)
        direction = {"斜前方": np.array([1., -1., .4]),
                     "正面": np.array([1., 0., .12]),
                     "侧面": np.array([0., -1., .12])}[view.value]
        client.camera.up_direction = (0., 0., 1.)
        client.camera.position = tuple(center + distance * direction / np.linalg.norm(direction))
        client.camera.look_at = tuple(center)

    @server.on_client_connect
    def on_connect(client):
        with lock:
            set_camera(client)

    @select.on_update
    def on_select(_):
        with lock:
            state.update(frame=0., changed=True, camera=True)
            frame.value = 0
            frame.max = len(clips[select.value][0]) - 1

    @frame.on_update
    def on_seek(event):
        if event.client is not None:
            with lock:
                playing.value = False
                state.update(frame=float(frame.value), changed=True)

    def step(amount):
        with lock:
            playing.value = False
            state.update(frame=float(np.clip(int(state["frame"]) + amount, 0, frame.max)), changed=True)

    previous.on_click(lambda _: step(-1))
    following.on_click(lambda _: step(1))

    @view.on_update
    @reset_camera.on_click
    def on_camera(_):
        with lock:
            state["camera"] = True

    @collisions.on_update
    def on_collisions(_):
        with lock:
            scene.geom_groups_visible[3] = collisions.value
            state["changed"] = True

    last_time = time.monotonic()
    last_draw = None
    print(f"Viewer ready: http://{args.host}:{server.get_port()} · {len(clips)} clips · PID {os.getpid()}", flush=True)
    try:
        while True:
            now = time.monotonic()
            dt, last_time = min(now - last_time, .1), now
            with lock:
                qpos, fps, desc = clips[select.value]
                if playing.value:
                    state["frame"] += dt * fps * float(speed.value[:-1])
                    if state["frame"] >= len(qpos):
                        if loop.value:
                            state["frame"] %= len(qpos)
                        else:
                            state["frame"] = float(len(qpos) - 1)
                            playing.value = False
                i = min(int(state["frame"]), len(qpos) - 1)
                if last_draw != (select.value, i) or state["changed"]:
                    data.qpos[:] = qpos[i]
                    mujoco.mj_kinematics(model, data)
                    with server.atomic():
                        scene.update_from_mjdata(data)
                        frame.value = i
                        description.content = desc
                        left, right = [foot_bottom(model, data, ids) * 1000 for ids in feet]
                        status.content = (
                            f"**{i/fps:.2f} / {(len(qpos)-1)/fps:.2f} 秒** · 第 {i+1}/{len(qpos)} 帧\n\n"
                            f"身体根部高度：{qpos[i,2]:.3f} m\n\n"
                            f"足底高度：左 {left:+.1f} mm / 右 {right:+.1f} mm")
                    last_draw, state["changed"] = (select.value, i), False
                if state["camera"]:
                    for client in server.get_clients().values():
                        set_camera(client)
                    state["camera"] = False
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
