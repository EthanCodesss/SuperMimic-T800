"""Synchronized 3D comparison of reference poses and saved T800 policy rollouts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from supermimic.robots.t800 import REPO_ROOT, get_spec
from tracking_replay import reference_frame_indices
from view_motions import load_clip

RUN = REPO_ROOT / "reports/training/t800_walk2_policy_ieee_stage1"
REPORTS = {
    "最终 · 2,000 次更新": "final.json",
    "阶段 · 1,984 次更新": "eval_ckpt_0005000.json",
    "阶段 · 1,484 次更新（偏差较大）": "eval_ckpt_0003750.json",
}
COLORS = ((55, 160, 235), (245, 130, 45))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    base = get_spec()
    single = base.compile()
    reference, fps = load_clip(REPO_ROOT / "data/npz/t800_single/walk2_steady.npz", single.nq)
    records = {}
    for label, filename in REPORTS.items():
        report_path = RUN / filename
        report = json.loads(report_path.read_text())
        physical, recorded_fps = load_clip(report_path.with_suffix(".trajectory.npz"), single.nq)
        if recorded_fps != fps:
            raise ValueError("Reference and physical recording must use the same sampling frequency")
        with np.load(report_path.with_suffix(".trajectory.npz")) as trajectory:
            indices, timing = reference_frame_indices(physical, reference, trajectory["root_error_m"])
        records[label] = (physical, reference[indices], report, timing)

    combined = mujoco.MjSpec()
    combined.option.integrator = base.option.integrator
    combined.option.tolerance = base.option.tolerance
    for name in ("reference", "policy"):
        combined.attach(get_spec(), prefix=name + "/", frame=combined.worldbody.add_frame())
    model = combined.compile()
    assert model.nq == 2 * single.nq
    for i in range(model.ngeom):
        name = model.body(model.geom_bodyid[i]).name
        color = COLORS[int(name.startswith("policy/"))]
        model.geom_matid[i] = -1
        model.geom_rgba[i] = [*(np.array(color) / 255), 1.]
    data = mujoco.MjData(model)
    server = viser.ViserServer(host=args.host, port=args.port, label="T800 · 参考与策略对比")
    server.gui.configure_theme(show_share_button=False, brand_color=COLORS[0])
    server.scene.set_up_direction("+z")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    scene.geom_groups_visible[3] = False
    server.scene.add_grid("/ground", width=20., height=20., cell_size=.2,
                          section_size=1., plane="xy", plane_opacity=.7)
    server.gui.add_markdown("## 参考 vs. 策略\n**蓝色：参考动作 · 橙色：实际策略轨迹**\n\n"
                            "同一条 3.96 秒行走转身动作；同步播放。拖动旋转，滚轮缩放。")
    select = server.gui.add_dropdown("检查点", options=tuple(records), initial_value=next(iter(records)))
    layout = server.gui.add_dropdown("对比方式", options=("并排播放", "同坐标叠加"))
    description = server.gui.add_markdown("")
    playing = server.gui.add_checkbox("播放", initial_value=True)
    loop = server.gui.add_checkbox("循环", initial_value=True)
    speed = server.gui.add_dropdown("播放速度", options=("0.25×", "0.5×", "1×"), initial_value="0.5×")
    frame = server.gui.add_slider("帧", min=0, max=len(records[select.value][0])-1, step=1, initial_value=0)
    previous = server.gui.add_button("上一帧")
    following = server.gui.add_button("下一帧")
    restart = server.gui.add_button("回到开头")
    paths_visible = server.gui.add_checkbox("显示根部行走路线", initial_value=True)
    view = server.gui.add_dropdown("视角", options=("斜前方", "正面", "俯视"))
    camera_reset = server.gui.add_button("重置视角")
    status = server.gui.add_markdown("")
    server.gui.add_markdown("地面小格 **20 厘米**。并排模式仅将两组画面横向移开，"
                            "误差始终按原始坐标计算；叠加模式保留真实位置偏差。\n\n"
                            "橙色回放来自策略在物理仿真中的实际记录。蓝色仅为参考姿态。"
                            "按评估目标帧同步，循环末尾会跳回开头。")
    labels = [server.scene.add_label(f"/label/{j}", text=text, position=(0, 0, 2.))
              for j, text in enumerate(("参考动作", "策略 · 最终 2,000 次更新"))]
    lines = []
    lock = threading.RLock()
    state = {"frame": 0., "changed": True, "paths": True, "camera": True}

    def offsets():
        if layout.value == "并排播放":
            return np.array([[-1.4, 0., 0.], [1.4, 0., 0.]])
        return np.zeros((2, 3))

    def set_camera(client):
        actual, target, _, _ = records[select.value]
        off = offsets()
        points = np.concatenate([target[:, :3] + off[0], actual[:, :3] + off[1]])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        center[2] = .75
        distance = max(4.8, float(np.ptp(points[:, :2], axis=0).max()) * 1.1 + 2.2)
        direction = {"斜前方": [.3, -1., .55], "正面": [0., -1., .3], "俯视": [0., -.05, 1.]}[view.value]
        direction = np.asarray(direction)
        client.camera.up_direction = (0., 1., 0.) if view.value == "俯视" else (0., 0., 1.)
        client.camera.position = tuple(center + distance * direction / np.linalg.norm(direction))
        client.camera.look_at = tuple(center)

    @server.on_client_connect
    def on_connect(client):
        with lock:
            set_camera(client)

    @select.on_update
    def on_select(_):
        with lock:
            state.update(frame=0., changed=True, paths=True, camera=True)
            frame.value = 0
            frame.max = len(records[select.value][0]) - 1

    @layout.on_update
    def on_layout(_):
        with lock:
            state.update(changed=True, paths=True, camera=True)

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

    @restart.on_click
    def on_restart(_):
        with lock:
            state.update(frame=0., changed=True)

    @paths_visible.on_update
    def on_paths(_):
        with lock:
            for line in lines:
                line.visible = paths_visible.value

    @view.on_update
    @camera_reset.on_click
    def on_camera(_):
        with lock:
            state["camera"] = True

    print(f"Viewer ready: http://{args.host}:{server.get_port()} · PID {os.getpid()}", flush=True)
    (RUN / "comparison-viewer.json").write_text(json.dumps({
        "url": f"http://{args.host}:{server.get_port()}", "pid": os.getpid(),
        "source": "saved deterministic physical evaluations; no new training or simulation",
        "records": {key: value[3] for key, value in records.items()},
    }, indent=2) + "\n")
    last_time, last_draw = time.monotonic(), None
    try:
        while True:
            now = time.monotonic()
            dt, last_time = min(now - last_time, .1), now
            with lock:
                actual, target, report, timing = records[select.value]
                off = offsets()
                if playing.value:
                    state["frame"] += dt * fps * float(speed.value[:-1])
                    if state["frame"] >= len(actual):
                        if loop.value:
                            state["frame"] %= len(actual)
                        else:
                            state["frame"] = float(len(actual) - 1)
                            playing.value = False
                i = min(int(state["frame"]), len(actual)-1)
                if state["paths"]:
                    for line in lines:
                        line.remove()
                    lines.clear()
                    for j, trajectory in enumerate((target, actual)):
                        xyz = trajectory[:, :3].copy() + off[j]
                        xyz[:, 2] = .012
                        lines.append(server.scene.add_line_segments(f"/path/{j}",
                            points=np.stack([xyz[:-1], xyz[1:]], axis=1), colors=COLORS[j],
                            thickness=.016, visible=paths_visible.value))
                    state["paths"] = False
                if last_draw != (select.value, i) or state["changed"]:
                    data.qpos[:single.nq] = target[i]
                    data.qpos[single.nq:] = actual[i]
                    data.qpos[:3] += off[0]
                    data.qpos[single.nq:single.nq+3] += off[1]
                    mujoco.mj_kinematics(model, data)
                    error = actual[i, :3] - target[i, :3]
                    with server.atomic():
                        scene.update_from_mjdata(data)
                        frame.value = i
                        for j, q in enumerate((target, actual)):
                            labels[j].position = tuple(q[i, :3] + off[j] + [0., 0., 1.05])
                        labels[1].text = f"策略 · {report['checkpoint_updates']:,} 次更新"
                        description.content = (
                            f"**{select.value}** · 整段完成\n\n"
                            f"评估平均根位置误差 **{report['mean_root_position_error_m']:.3f} m** · "
                            f"归一化回报 **{report['normalized_return']:.3f}**")
                        status.content = (
                            f"**{i/fps:.2f} / {(len(actual)-1)/fps:.2f} 秒** · 第 {i+1}/{len(actual)} 帧\n\n"
                            f"当前根位置误差 **{np.linalg.norm(error):.3f} m**\n\n"
                            f"水平 {np.linalg.norm(error[:2]):.3f} m · 高度 {abs(error[2]):.3f} m")
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
