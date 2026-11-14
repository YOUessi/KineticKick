"""Compare two full-speed GMT simulations to one original human reference in a shared camera."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from v161_evaluate import DEFAULT_URDF
from v163_compare_motion import (
    AQUA,
    DARK,
    MUTED,
    RULE,
    WHITE,
    camera,
    digest,
    draw_stats,
    goal_mesh,
    load_geometry,
    load_run,
    matrices,
    panel_background,
    scene,
    source_pose,
    write_text,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("before", "after", "reference", "clearance", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--after-label", default="改进后  ·  支撑反馈与全身间隙修正")
    a = p.parse_args()
    runs = [load_run(a.before), load_run(a.after)]
    if a.output.exists():
        raise FileExistsError(a.output)
    source = digest(a.reference)
    registered = json.loads(a.clearance.read_text())
    if registered["input_sha256"] != source or registered["urdf_sha256"] != digest(
        DEFAULT_URDF
    ):
        raise ValueError("Reference transformation does not match source")
    if [r["trial"]["reference_sha256"] for r in runs] != [
        source,
        registered["output_sha256"],
    ]:
        raise ValueError("Reference digests disagree with run manifests")
    if any(
        not np.allclose(runs[0]["geometry"][k], runs[1]["geometry"][k])
        for k in runs[0]["geometry"]
    ):
        raise ValueError("Goal changed")
    if not np.allclose(
        runs[0]["trial"]["ball_initial_xyz_m"], runs[1]["trial"]["ball_initial_xyz_m"]
    ):
        raise ValueError("Ball changed")
    with np.load(a.reference, allow_pickle=False) as ref:
        rp = ref["body_pos_w"]
        rr = matrices(ref["body_quat_w"])
        rt = ref["time_s"]
        clip = ref["source_clip"].item()
        if any(r["names"] != ref["body_names"].tolist() for r in runs):
            raise ValueError("Bodies changed")
    for run, folder in zip(runs, [a.before, a.after]):
        score_file = folder / "support_score.json"
        if score_file.is_file():
            score = json.loads(score_file.read_text())
            if score["rollout_sha256"] != run["digests"]["rollout.npz"]:
                raise ValueError("Wrong support audit")
            run["tail"] = score["strict_standing_tail_s"]
    shapes = load_geometry(DEFAULT_URDF, runs[0]["names"])
    cam = camera()
    cam.scale = 176.0
    cr = camera()
    cr.scale = 136.0
    cv2.setNumThreads(1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name("." + a.output.stem + ".encoding.mp4")
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            "1280x720",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(7, 2800, 8):
            t = runs[0]["t"][i]
            ri = min(len(rt) - 1, round(t * 50))
            cam.center = np.r_[rp[ri, 0, :2], 0.56]
            bg = panel_background(cam, False)
            goal_mesh(bg, cam, runs[0]["geometry"])
            img = Image.new("RGB", (1280, 720), DARK)
            for j, r in enumerate(runs):
                img.paste(scene(r, i, shapes, cam, bg), (640 * j, 101))
            img.paste(source_pose(cr, shapes, rp[ri], rr[ri]), (541, 538))
            d = ImageDraw.Draw(img)
            write_text(
                d, (23, 4), "真人动作 " + clip + "  ·  全速跟踪与恢复", 33, WHITE, True
            )
            write_text(d, (1255, 15), f"{t:.2f} / 14.00 s", 24, AQUA, True, "ra")
            for y in (57, 532, 708):
                d.line((0, y, 1280, y), fill=RULE, width=2)
            d.line((640, 57, 640, 532), fill=RULE, width=2)
            write_text(d, (24, 62), "改进前  ·  GMT", 24, WHITE, True)
            write_text(
                d, (665, 62), a.after_label, 24, AQUA, True
            )
            for j, r in enumerate(runs):
                x = 24 if j == 0 else 778
                draw_stats(d, r, i, x, t >= rt[-1])
                # Display world error instead of a stationary ball's speed.
                if t <= 14 - r["tail"]:
                    d.rectangle((x, 649, x + 420, 686), fill=DARK)
                    err = np.linalg.norm(r["body"][i, 0] - rp[ri, 0])
                    label = (
                        f"骨盆相对参考偏差    {err * 100:.1f} cm"
                        if t <= rt[-1]
                        else "动作参考已经结束"
                    )
                    write_text(d, (x, 651), label, 19, MUTED)
            write_text(d, (641, 539), "原始重定向参考", 15, WHITE, True, "ma")
            write_text(d, (641, 676), f"源时间 {rt[ri]:.2f} s", 15, MUTED, True, "ma")
            d.line((20, 708, 20 + 1240 * t / 14.0, 708), fill=AQUA, width=3)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Encoder failure")
        os.link(tmp, a.output)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        tmp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "video_sha256": digest(a.output),
                "source_sha256": source,
                "before_after": [r["digests"] for r in runs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
