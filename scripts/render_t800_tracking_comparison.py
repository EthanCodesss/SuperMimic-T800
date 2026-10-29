"""Render paired, already-evaluated T800 states without re-running the policy.

Three synchronized panels share a camera and world coordinates: reference,
tracking, and a translucent reference over the tracked robot. Writes a full
MP4, a poster, provenance, and a local HTML player with synchronized error plots.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ['SUPERMIMIC_ROBOT'] = 't800'


def make_scene():
    import mujoco
    import numpy as np
    from supermimic.robots.t800 import get_spec, JOINT_NAMES
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.tolerance = 1e-10
    colors = ([.18, .52, .83, 1.], [.90, .39, .12, 1.])
    for prefix, group, color in zip(('ref_', 'sim_'), (1, 2), colors):
        child = get_spec()
        for geom in child.geoms:
            if geom.group != 3:
                geom.group = group
                geom.material = ''
                geom.rgba = color
        spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
    spec.add_texture(name='floor_grid', type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
        rgb1=[.87, .90, .93], rgb2=[.78, .82, .86])
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=256,
        rgb1=[.91, .94, .97], rgb2=[.78, .84, .90])
    material = spec.add_material(name='floor_material', texrepeat=[2, 2], texuniform=True,
                                 reflectance=0, specular=.1, shininess=.1)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'floor_grid'
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, .05], material='floor_material', group=0, contype=0, conaffinity=0)
    spec.worldbody.add_light(name='sun', pos=[0, -3, 6], dir=[.25, .3, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[.55, .55, .55], specular=[.12, .12, .12], castshadow=True)
    model = spec.compile()
    assert (model.nq, model.nv) == (64, 62)
    for prefix, offset in [('ref_', 0), ('sim_', 32)]:
        assert [model.joint(prefix+n).qposadr[0] for n in JOINT_NAMES] == list(range(offset+7, offset+32))
    model.vis.headlight.ambient[:] = [.42, .42, .42]
    model.vis.headlight.diffuse[:] = [.45, .45, .45]
    model.vis.headlight.specular[:] = [.1, .1, .1]
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 4
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 624
    model.vis.map.znear = .01
    return model, np.flatnonzero(model.geom_group == 1)


def write_player(out, data, evaluation, manifest):
    import numpy as np
    q, ref = data['qpos'], data['reference_qpos']
    body = np.sqrt(np.mean((q[:, 7:30]-ref[:, 7:30])**2, axis=-1))
    # Embed data so opening the HTML via file:// needs no fetch/server/CDN.
    samples = np.stack((data['time_s'], data['root_error_m'],
                        np.abs(data['yaw_error_deg']), body), axis=-1)
    payload = json.dumps(dict(samples=np.round(samples[::5], 5).tolist(),
        duration=len(q)/50, worst_root_time=float(data['time_s'][np.argmax(data['root_error_m'])]),
        metrics=dict(return_=evaluation['normalized_full_clip_return'],
        root=evaluation['mean_root_position_error_m'], yaw=evaluation['mean_abs_yaw_error_deg'],
        body=evaluation['non_head_joint_rmse_rad'])), separators=(',', ':'))
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T800 参考动作与跟踪结果</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#1d3046;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px 24px}h1{font-size:28px;margin:0 0 8px}p{margin:8px 0 18px}.muted{color:#52667b}video{display:block;width:100%;background:#111c29;border-radius:12px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:14px 0}button,select,a.download{font:inherit;padding:7px 12px;border:1px solid #ccd8e3;border-radius:7px;background:white;color:#174a79;cursor:pointer;text-decoration:none}button:hover{background:#dfeaf4}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.stat{background:white;padding:12px 16px;border-radius:9px}.stat strong{display:block;font-size:23px}.stat small{color:#566a7f}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}.plot{background:white;padding:14px;border-radius:10px}.plot h2{font-size:16px;margin:0 0 5px}canvas{width:100%;height:180px;cursor:crosshair}a{color:#165c93}.foot{font-size:14px;color:#52667b;margin-top:20px}@media(max-width:750px){.stats,.plots{grid-template-columns:1fr 1fr}main{padding:16px}.plots{grid-template-columns:1fr}}
</style><main><h1>T800：参考动作与实际跟踪</h1>
<p class="muted">8104 次更新 · anchor 观测 + 固定头部 · 完整 walk1，261.32 秒</p>
<video id="video" controls preload="metadata" poster="poster.png" playsinline src="comparison_full.mp4"></video>
<div class="toolbar"><button id="toggle">播放 / 暂停</button><button id="back">上一帧</button><button id="next">下一帧</button><label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label><span id="clock"></span><a class="download" href="comparison_full.mp4" download>下载完整视频</a></div>
<div class="toolbar" id="jumps"></div>
<p>左：<b style="color:#287cc0">蓝色参考动作</b>；中：<b style="color:#dc641f">橙色实际跟踪</b>；右：实际跟踪与半透明参考叠加。三个视图共用相机与世界坐标，跟随参考根位置移动，保留真实位姿误差。</p>
<div class="stats"><div class="stat"><small>完整动作 evaluation return</small><strong id="ret"></strong></div><div class="stat"><small>当前根位置误差</small><strong id="root"></strong></div><div class="stat"><small>当前 |yaw| 误差</small><strong id="yaw"></strong></div><div class="stat"><small>当前身体 23 关节 RMSE</small><strong id="body"></strong></div></div>
<div class="plots"><div class="plot"><h2>根位置误差（cm）· 点击曲线跳转</h2><canvas id="rootplot"></canvas></div><div class="plot"><h2>|yaw| 误差（°）· 点击曲线跳转</h2><canvas id="yawplot"></canvas></div></div>
<p class="foot">视频由最终 IEEE FP32 评估保存的真实仿真关节和根状态渲染；参考帧与实际状态逐帧配对。控制记录为 50 Hz，视频为 25 FPS、原速，无姿态插值。显示颜色仅用于区分机器人。头部通过 PD 跟踪默认角度，仍保留实际物理运动。当前数值按最近的 0.1 秒采样显示。<br><a href="../report.html">实验报告</a> · <a href="manifest.json">渲染来源与参数</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),v=document.getElementById('video'),S=D.samples;
document.getElementById('ret').textContent=D.metrics.return_.toFixed(4);
document.getElementById('toggle').onclick=()=>v.paused?v.play():v.pause();
document.getElementById('back').onclick=()=>{v.pause();v.currentTime=Math.max(0,v.currentTime-1/25)};
document.getElementById('next').onclick=()=>{v.pause();v.currentTime=Math.min(D.duration-.04,v.currentTime+1/25)};
document.getElementById('speed').onchange=e=>v.playbackRate=+e.target.value;
for(const [label,t] of [['开头',0],['60 秒',60],['120 秒',120],['180 秒',180],['240 秒',240],['最大根位置误差',D.worst_root_time]]){const b=document.createElement('button');b.textContent=label;b.onclick=()=>v.currentTime=t;document.getElementById('jumps').appendChild(b)}
function plot(id,col,scale){const c=document.getElementById(id),r=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=r.width*dpr;c.height=r.height*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=r.width,h=r.height,L=44,R=w-12,T=10,B=h-28,max=Math.max(...S.map(s=>s[col]*scale))*1.1;ctx.font='12px system-ui';ctx.lineWidth=1;for(let k=0;k<=4;k++){const y=B-(B-T)*k/4;ctx.strokeStyle='#e4eaf0';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(R,y);ctx.stroke();ctx.fillStyle='#53677c';ctx.fillText((max*k/4).toFixed(1),2,y+4)}ctx.beginPath();ctx.strokeStyle=col===1?'#1a78b5':'#bc6126';ctx.lineWidth=1.5;S.forEach((s,i)=>{const x=L+(R-L)*s[0]/D.duration,y=B-(B-T)*s[col]*scale/max;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.strokeStyle='#273b50';ctx.beginPath();const x=L+(R-L)*v.currentTime/D.duration;ctx.moveTo(x,T);ctx.lineTo(x,B);ctx.stroke();ctx.fillStyle='#53677c';[0,60,120,180,240].forEach(t=>ctx.fillText(t+'s',L+(R-L)*t/D.duration-8,h-7));c.onclick=e=>v.currentTime=Math.max(0,Math.min(D.duration-.04,(e.offsetX-L)/(R-L)*D.duration))}
function update(){const s=S[Math.min(S.length-1,Math.round(v.currentTime*10))];document.getElementById('clock').textContent=v.currentTime.toFixed(2)+' / '+D.duration.toFixed(2)+' 秒';document.getElementById('root').textContent=(s[1]*100).toFixed(1)+' cm';document.getElementById('yaw').textContent=s[2].toFixed(2)+'°';document.getElementById('body').textContent=s[3].toFixed(3)+' rad';plot('rootplot',1,100);plot('yawplot',2,1)}
v.addEventListener('timeupdate',update);v.addEventListener('loadedmetadata',update);window.addEventListener('resize',update);update();
</script></main></html>'''
    (out / 'index.html').write_text(doc.replace('__DATA__', payload))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', type=Path, default=ROOT/'reports/training/t800_walk1_anchor_headfixed_8104/eval_ckpt_u0008104.trajectory.npz')
    p.add_argument('--evaluation', type=Path, default=ROOT/'reports/training/t800_walk1_anchor_headfixed_8104/eval_ckpt_u0008104.json')
    p.add_argument('--out', type=Path, default=ROOT/'reports/t800_anchor_head_study/visualization')
    p.add_argument('--seconds', type=float, help='Optional preview duration; omit for complete clip')
    args = p.parse_args()
    import mujoco
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    args.out.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.trajectory))
    evaluation = json.loads(args.evaluation.read_text())
    q, ref = data['qpos'], data['reference_qpos']
    assert q.shape == ref.shape == (evaluation['steps'], 32)
    assert float(data['fps'][0]) == 50
    assert np.allclose(np.linalg.norm(q[:,:3]-ref[:,:3],axis=-1), data['root_error_m'], atol=2e-5)
    assert np.allclose(np.sqrt(np.mean((q[:,7:]-ref[:,7:])**2)), evaluation['joint_position_rmse_rad'], atol=1e-6)
    total = min(len(q), round(args.seconds*50)) if args.seconds else len(q)
    fps, stride, width, height = 25, 2, 1920, 768
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 26)
    small = ImageFont.truetype(font_path, 19)
    model, ref_geoms = make_scene()
    sim = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=624, width=640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.25, 135., -16.
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.sitegroup[:] = 0
    # Never integrate dynamics here: the recorded states are authoritative.
    output = args.out / 'comparison_full.mp4'
    if output.exists():
        raise FileExistsError(output)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', '-threads', '4', str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    base = Image.new('RGB', (width, height), '#f4f7fb')
    draw = ImageDraw.Draw(base)
    for x, title, subtitle, color in [
        (0, '参考动作', 'T800 重定向目标', '#217abc'),
        (640, '实际跟踪', '8104 次更新 · anchor + 固定头部', '#ca5b20'),
        (1280, '同坐标叠加', '蓝色半透明参考 / 橙色实际跟踪', '#283e54')]:
        draw.text((x+22, 6), title, font=font, fill=color)
        draw.text((x+22, 44), subtitle, font=small, fill='#52667b')
    start = time.monotonic()
    count = 0
    target = ref[0,:2].copy()
    try:
        for i in range(0, total, stride):
            sim.qpos[:32], sim.qpos[32:] = ref[i], q[i]
            mujoco.mj_forward(model, sim)
            target = .88*target + .12*ref[i,:2]
            cam.lookat[:] = [*target, .90]
            frame = base.copy()
            for panel, visible, alpha in [(0,(1,),1.), (1,(2,),1.), (2,(1,2),.30)]:
                opt.geomgroup[1:3] = 0
                for group in visible:
                    opt.geomgroup[group] = 1
                model.geom_rgba[ref_geoms,3] = alpha
                renderer.update_scene(sim, camera=cam, scene_option=opt)
                frame.paste(Image.fromarray(renderer.render()), (panel*640,80))
            body_rmse = float(np.sqrt(np.mean((q[i,7:30]-ref[i,7:30])**2)))
            draw = ImageDraw.Draw(frame)
            for x in [640,1280]:
                draw.line((x,0,x,height),fill='#d6e0ea',width=2)
            draw.text((22,714), f't = {i/50:6.2f} / {len(q)/50:.2f} s   |   原速 1×', font=small, fill='#263e55')
            draw.text((640+22,714), f'根位置 {data["root_error_m"][i]*100:.1f} cm   |   yaw {abs(data["yaw_error_deg"][i]):.2f}°', font=small, fill='#263e55')
            draw.text((1280+22,714), f'身体关节 RMSE {body_rmse:.3f} rad   |   eval {evaluation["normalized_full_clip_return"]:.4f}', font=small, fill='#263e55')
            encoder.stdin.write(np.asarray(frame).tobytes())
            if count == 0 or i == 600:
                frame.save(args.out/'poster.png')
            count += 1
            if count % 250 == 0:
                print(json.dumps(dict(rendered=count, total=(total+1)//2,
                    elapsed_seconds=round(time.monotonic()-start,1))), flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoder failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    finally:
        renderer.close()
    manifest = dict(status='complete', source_trajectory=str(args.trajectory),
        source_trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        evaluation=str(args.evaluation), checkpoint=evaluation['checkpoint'], updates=evaluation['checkpoint_updates'],
        render_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frames=count, video_fps=fps, video_duration_s=count/fps, source_frames=len(q),
        source_fps=50, resolution=[width,height], frame_stride=stride,
        coordinate_alignment='Original shared world coordinates, reference-centered camera; no pose/heading alignment',
        replay='Saved real simulator qpos paired with saved command reference_qpos; no policy or dynamics rerun',
        colors=dict(reference='blue', tracking='orange', overlay_reference_alpha=.30),
        elapsed_seconds=time.monotonic()-start, file_bytes=output.stat().st_size)
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if not args.seconds:
        write_player(args.out,data,evaluation,manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
