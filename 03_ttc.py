"""
How early can we tell when the stairs arrive?

Run this inside a session folder, after 02_analyze.py:
    python3 03_ttc.py

Reads  : features_final.csv
Writes : ttc_analysis.png, ttc_summary.png

Time-to-contact is just distance divided by closing speed. The catch is
getting a usable speed: differentiating the raw distance amplifies the step
bounce badly, so speed comes from a least-squares fit over the last 1.5 s
instead.

Known failure mode: people slow down as they reach the stairs. Closing speed
peaked around 3.9 s before arrival (0.49 m/s) and had dropped to 0.095 m/s by
2.3 s. Since speed is the denominator, TTC blows up in that last stretch.
Treat 3-5 s out as the usable window and fall back to a plain distance
threshold closer in.
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW = 6.0      # seconds of history to plot before each event
FIT_WIN = 1.5     # regression window for closing speed
MIN_SPEED = 0.05  # m/s; below this we're not meaningfully approaching anything


def load():
    rows = sorted(csv.DictReader(open(os.path.join(HERE, "features_final.csv"))),
                  key=lambda r: float(r["t"]))
    t = np.array([float(r["t"]) for r in rows])
    d = np.array([float(r["p05"]) for r in rows])
    state = [r["state"] for r in rows]
    return t, d, state


def smooth_distance(t, d):
    """One-second moving average. Step bounce sits around 1-2 Hz, so this
    flattens it while leaving the approach trend alone."""
    fps = 1 / np.median(np.diff(t))
    win = max(int(fps * 1.0), 1)
    filled = np.where(np.isfinite(d), d, np.nanmedian(d))
    return np.convolve(filled, np.ones(win) / win, mode="same")


def find_events(t, state):
    """First frame of each stair interval."""
    events = []
    for i in range(1, len(state)):
        if state[i] in ("stair_up", "stair_down") and state[i - 1] != state[i]:
            events.append((t[i], state[i]))
    return events


def closing_speed(tt, dd, i, fit_win):
    m = (tt >= tt[i] - fit_win) & (tt <= tt[i])
    if m.sum() < 4:
        return np.nan
    return -np.polyfit(tt[m], dd[m], 1)[0]   # negative slope = getting closer


def main():
    t, d_raw, state = load()
    d = smooth_distance(t, d_raw)

    events = find_events(t, state)
    print(f"{len(events)} stair events: {events}")
    if not events:
        print("No stair labels found. Check labels.csv for stair_up / stair_down.")
        return

    fig, axes = plt.subplots(len(events), 2, figsize=(13, 3.1 * len(events)))
    if len(events) == 1:
        axes = axes.reshape(1, 2)

    errors_by_lead = {}
    for row, (t_evt, label) in enumerate(events):
        m = (t >= t_evt - WINDOW) & (t <= t_evt + 0.5)
        tt, dd = t[m], d[m]
        if len(tt) < 10:
            continue

        ttc = np.full(len(tt), np.nan)
        for i in range(len(tt)):
            v = closing_speed(tt, dd, i, FIT_WIN)
            if v and v > MIN_SPEED:
                ttc[i] = dd[i] / v
        err = (tt + ttc) - t_evt      # predicted arrival minus actual

        ax = axes[row, 0]
        ax.plot(tt - t_evt, dd, color="black", lw=1.6)
        ax.axvline(0, color="red", ls="--")
        ax.set_ylabel("distance (m)")
        ax.set_xlabel("time to event (s)")
        ax.set_title(f"{label} @ {t_evt:.1f}s")
        ax.grid(alpha=0.25)

        ax = axes[row, 1]
        lead = t_evt - tt
        ok = np.isfinite(err) & (np.abs(err) < 15)
        ax.plot(lead[ok], err[ok], ".", color="tab:blue", ms=5)
        ax.axhline(0, color="red", ls="--")
        ax.axhspan(-1, 1, color="green", alpha=0.12)
        ax.invert_xaxis()
        ax.set_ylim(-6, 6)
        ax.set_ylabel("arrival error (s)")
        ax.set_xlabel("lead time (s)")
        ax.grid(alpha=0.25)

        for bucket in (5, 4, 3, 2, 1):
            sel = (lead >= bucket - 0.5) & (lead < bucket + 0.5) & ok
            if sel.sum():
                errors_by_lead.setdefault(bucket, []).extend(err[sel].tolist())

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "ttc_analysis.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    buckets = sorted(errors_by_lead)
    mae = [np.mean(np.abs(errors_by_lead[b])) for b in buckets]
    ax.plot(buckets, mae, marker="o", color="tab:blue", lw=2, ms=8)
    for b, m in zip(buckets, mae):
        ax.annotate(f"n={len(errors_by_lead[b])}", (b, m),
                    textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    ax.invert_xaxis()
    ax.set_xlabel("lead time (s)")
    ax.set_ylabel("mean absolute error (s)")
    ax.set_title(f"Arrival time prediction accuracy (n={len(events)} events)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "ttc_summary.png"), dpi=140)

    print(f"\n{'lead(s)':>9}{'MAE(s)':>10}{'median(s)':>11}{'n':>6}")
    for b in buckets:
        e = np.abs(errors_by_lead[b])
        print(f"{b:9d}{e.mean():10.2f}{np.median(e):11.2f}{len(e):6d}")

    print("\nwrote ttc_analysis.png and ttc_summary.png")
    print("Expect the last ~2 s to look bad; see the note at the top of this file.")


if __name__ == "__main__":
    main()
