"""
Extract gait and terrain signals from one session.

Run this inside a session folder, after 01_make_labels.py:
    python3 02_analyze.py

Reads  : frames.csv, depth_*.f32, uw_*.jpg, labels.csv
Writes : features_final.csv, analyze_final.png

Two independent signals come out of here:

  Gait    - optical flow on the downward (mirror) view. The phone rides on the
            waist so the whole frame shakes with each step; subtracting the
            median flow leaves the legs, which move differently from everything
            else. Peaks in that signal are steps.

  Terrain - a vertical slice through the middle of the forward depth map,
            giving a "distance profile". Flat ground, stairs up and stairs down
            each produce a distinctly shaped profile.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))

# Region of the downward frame where the legs actually appear, in normalised
# coordinates. Re-measure this if the mirror or mount moves — everything
# downstream depends on it.
ROI_X0, ROI_X1 = 0.18, 0.81
ROI_Y0, ROI_Y1 = 0.38, 0.62

DOWNSCALE = 4    # shrink before optical flow; full res is far slower, no better
STEP_GAIT = 1    # use every downward frame — dropping frames loses step peaks
STEP_DEPTH = 2   # depth is smoother, every other frame is plenty


def load_frames():
    depth, uw = [], []
    with open(os.path.join(HERE, "frames.csv")) as f:
        for r in csv.DictReader(f):
            rec = (r["filename"], float(r["hostTime"]), int(r["width"]), int(r["height"]))
            (depth if r["kind"] == "depth" else uw).append(rec)
    depth.sort(key=lambda x: x[1])
    uw.sort(key=lambda x: x[1])
    return depth, uw


def load_segments(t0, t_end):
    """A label marks a transition, not an instant. It stays in effect until the
    next one, so turn the list into intervals."""
    raw = []
    path = os.path.join(HERE, "labels.csv")
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                raw.append((float(r["hostTime"]) - t0, r["label"]))
    raw.sort()
    return [(t, raw[i + 1][0] if i + 1 < len(raw) else t_end, lb)
            for i, (t, lb) in enumerate(raw)]


def roi_gray(fn):
    im = cv2.imread(os.path.join(HERE, fn), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    r = im[int(ROI_Y0 * h):int(ROI_Y1 * h), int(ROI_X0 * w):int(ROI_X1 * w)]
    return cv2.resize(r, (max(r.shape[1] // DOWNSCALE, 8),
                          max(r.shape[0] // DOWNSCALE, 8)))


def gait_signal(uw, t0):
    print("computing gait signal")
    frames = uw[::STEP_GAIT]
    times, motion = [], []
    prev = roi_gray(frames[0][0])
    for i, (fn, t, _w, _h) in enumerate(frames[1:]):
        cur = roi_gray(fn)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Median = whatever the whole frame is doing, i.e. torso sway.
        motion.append(np.clip(mag - np.median(mag), 0, None).mean())
        times.append(t - t0)
        prev = cur
        if i % 500 == 0:
            print(f"  {i * STEP_GAIT}/{len(uw)}")
    return np.array(times), np.array(motion)


def terrain_features(depth, t0, state_at):
    print("computing terrain features")
    rows = []
    for i, (fn, t, w, h) in enumerate(depth[::STEP_DEPTH]):
        d = np.fromfile(os.path.join(HERE, fn), dtype=np.float32).reshape(h, w)
        d = np.where(np.isfinite(d) & (d > 0.15) & (d < 8), d, np.nan)
        band = d[:, w // 3:2 * w // 3]           # middle third = direction of travel
        prof = np.nanmedian(band, axis=1)
        v = prof[np.isfinite(prof)]
        if len(v) < 20:
            continue
        diffs = np.diff(v)
        n = len(v)
        rows.append({
            "t": t - t0,
            "p05": np.nanpercentile(v, 5),        # nearest thing in front
            "p50": np.nanmedian(v),
            # Top third minus bottom third of the profile. This is the single
            # most useful terrain feature we found — it flips sign between
            # climbing and descending stairs.
            "top_minus_bot": np.nanmean(v[:n // 3]) - np.nanmean(v[-n // 3:]),
            "max_jump": np.max(np.abs(diffs)),    # step edges / drop-offs
            "state": state_at(t - t0),
        })
        if i % 300 == 0:
            print(f"  {i * STEP_DEPTH}/{len(depth)}")
    return rows


def main():
    depth, uw = load_frames()
    t0 = min(depth[0][1], uw[0][1])
    t_end = max(depth[-1][1], uw[-1][1]) - t0
    print(f"{len(depth)} depth frames, {len(uw)} downward frames, {t_end:.0f}s")

    segments = load_segments(t0, t_end)
    print(f"{len(segments)} label intervals")

    def state_at(t):
        for s, e, lb in segments:
            if s <= t < e:
                return lb
        return ""

    t_gait, motion = gait_signal(uw, t0)
    fps = len(t_gait) / (t_gait[-1] - t_gait[0])
    smooth = np.convolve(motion, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth,
                          distance=max(int(fps * 0.3), 2),
                          prominence=smooth.std() * 0.5)
    print(f"  {fps:.1f} fps, {len(peaks)} steps")

    rows = terrain_features(depth, t0, state_at)
    with open(os.path.join(HERE, "features_final.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote features_final.csv")

    colours = {"stop": "#F4B6B6", "floor": "#C9E0F5", "turn": "#F7D9A8",
               "slope_up": "#C8E6C0", "slope_down": "#EDE7A8",
               "stair_up": "#9FD8A0", "stair_down": "#F0C77E",
               "door": "#E3C9EC", "start": "white", "end": "white"}

    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for a in ax:
        for s, e, lb in segments:
            a.axvspan(s, e, color=colours.get(lb, "#EEEEEE"), alpha=0.6, lw=0)

    ax[0].plot(t_gait, motion, color="black", lw=0.6)
    ax[0].plot(t_gait[peaks], motion[peaks], "r.", ms=3, label=f"{len(peaks)} steps")
    ax[0].set_ylabel("leg motion")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Gait - optical flow in mirror ROI")

    ax[1].plot([r["t"] for r in rows], [r["p05"] for r in rows], color="black", lw=0.8)
    ax[1].set_ylabel("m")
    ax[1].set_xlabel("s")
    ax[1].set_title("Terrain - nearest distance ahead")

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "analyze_final.png"), dpi=110)
    print("wrote analyze_final.png")

    grouped = defaultdict(list)
    for r in rows:
        if r["state"]:
            grouped[r["state"]].append(r)

    print(f"\n{'state':12}{'n':>6}{'top_minus_bot':>16}{'nearest(m)':>12}")
    for lb in sorted(grouped):
        g = grouped[lb]
        print(f"{lb:12}{len(g):6d}"
              f"{np.mean([x['top_minus_bot'] for x in g]):16.2f}"
              f"{np.mean([x['p05'] for x in g]):12.2f}")

    print("\nIf top_minus_bot separates across states, terrain is distinguishable.")


if __name__ == "__main__":
    main()
