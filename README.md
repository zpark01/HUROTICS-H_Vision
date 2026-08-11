# H-Vision

Smartphone-based terrain and gait sensing for walking assistance.

An iPhone worn at the waist watches two things at once: the ground ahead, using
the LiDAR depth camera, and the wearer's own legs, using the ultra-wide camera
reflected through a 45° prism mirror. The goal is to know that stairs are coming
*before* a foot lands on them, so an assistive device can prepare instead of react.

---

## Why

H-Medi currently relies on IMU data alone. An IMU only reports motion that has
already happened, so terrain is discovered on contact and assistance is always a
beat late. There is also no convenient way to annotate walking data — labelling
has meant operating a remote while walking.

This project tests whether a phone can cover both gaps at once, with no extra
hardware: predict terrain ahead, observe gait, and label the recording by voice.

---

## What's in this package

| File | Purpose |
|---|---|
| `ContentView.swift` | iOS logging app — full source |
| `01_make_labels.py` | Audio → timestamped motion labels |
| `02_analyze.py` | Gait and terrain signal extraction |
| `03_ttc.py` | Stair arrival-time prediction |
| `H-Vision_인수인계.pdf` | Full handover document (Korean) |

The PDF is the authoritative reference — setup steps, sensor placement
rationale, known limitations and roadmap. This README is the short version.

---

## Hardware

| Item | Notes |
|---|---|
| iPhone 15 Pro or newer | LiDAR required. Developed on 16 Pro |
| Prism mirror | 3D printed. Must cover the ultra-wide lens only |
| Belt mount | Waist-centred, cameras facing forward |
| Mac | Xcode plus the analysis scripts |
| Bluetooth earbuds | Recommended — mic near the mouth, much better labels |

**The forward camera deliberately does not use a mirror.** Reflected light
corrupts the LiDAR depth estimate. Only the ultra-wide, which contributes RGB
and no depth, goes through the prism.

---

## Setup

Create a SwiftUI project in Xcode, replace `ContentView.swift` with the one here,
then add these under TARGETS → Info:

```
Privacy - Camera Usage Description          <any text>
Privacy - Microphone Usage Description      <any text>
Application supports iTunes file sharing    YES
Supports opening documents in place         YES
```

Enable Developer Mode on the phone, build to the device, and trust the developer
profile under Settings → General → VPN & Device Management.

> A free Apple ID signs the app for 7 days. When it stops launching, just build
> again. Do not delete the app before copying sessions off it — the recordings
> live inside the app container.

Python side:

```bash
pip3 install numpy opencv-python matplotlib scipy openai-whisper imageio-ffmpeg
```

ffmpeg ships with `imageio-ffmpeg`; the label script puts it on PATH itself, so
Homebrew isn't needed.

---

## Recording

1. Connect earbuds, open the app, mount the phone.
2. Adjust the focus slider until the legs in the mirror look sharp.
3. Check the depth coverage readout on the forward view.
4. Hit record, then walk.
5. Speak a label just before entering each terrain type.

| Say | Label |
|---|---|
| 시작 / 끝 | `start` / `end` |
| 바닥 | `floor` |
| 계단 위 / 계단 아래 | `stair_up` / `stair_down` |
| 오르막 / 내리막 | `slope_up` / `slope_down` |
| 정지 / 회전 / 문 | `stop` / `turn` / `door` |

Keep it to the keyword, leave a beat between labels, and speak before entering
the terrain rather than during it. A label stays in effect until the next one.

Sessions land in the app's Documents folder. Pull them off via Finder →
iPhone → Files.

### What gets written

```
session_<timestamp>/
  frames.csv          index: kind, filename, hostTime, resolution
  depth_XXXXXX.f32    forward depth, raw float32, metres
  uw_XXXXXX.jpg       downward RGB through the mirror
  imu.csv             100 Hz, includes the gravity vector
  audio.m4a           voice track
  audio_start.txt     hostTime of audio position zero
  camera_info.txt     depth resolution, FOV, intrinsics
```

Everything shares one `hostTime` clock, which is what makes the streams line up
later — including, eventually, leg IMU data from H-Medi.

---

## Analysis

Copy the three scripts into a session folder and run them in order:

```bash
python3 01_make_labels.py    # → labels.csv
python3 02_analyze.py        # → features_final.csv, analyze_final.png
python3 03_ttc.py            # → ttc_analysis.png, ttc_summary.png
```

**Labelling.** Whisper large-v3 with word timestamps, then keyword matching.
`condition_on_previous_text=False` is not optional — with it enabled, Whisper
locks onto its own previous output when short commands repeat and transcribes
entire stair sections as "바닥".

**Gait.** Optical flow over the mirror ROI. The whole frame shakes with each
step, so subtracting the median flow isolates the legs. Peak spacing gives
cadence.

**Terrain.** A vertical slice down the centre of the depth map produces a
distance profile whose shape depends on terrain. The most discriminative feature
is `top_minus_bot` — the top third of the profile minus the bottom third — which
changes sign between climbing and descending stairs.

**Arrival prediction.** TTC = distance ÷ closing speed, with speed from a 1.5 s
regression rather than a derivative.

---

## Results so far

| Metric | Value |
|---|---|
| Arrival prediction, 5 s out | 0.43 s error |
| Arrival prediction, 4 s out | 0.95 s error |
| Arrival prediction, 3 s out | 1.42 s error |
| Terrain separation, flat vs stairs up | effect size 4.11 |
| Terrain separation, slope up vs down | effect size 2.68 |
| Cadence | 102 steps/min, SNR 19.6 |

Effect sizes above 0.8 are normally read as a clear separation.

---

## Limitations

**TTC breaks down in the final approach.** People decelerate near stairs.
Closing speed peaked at 0.49 m/s roughly 3.9 s out and fell to 0.095 m/s by
2.3 s. Speed is the denominator, so the estimate diverges. Usable window is
3–5 s out; something distance-based is needed closer in.

**Not enough stair events.** Long, straight approaches are rare in the current
recordings, which is what the prediction numbers rest on. This is the main
bottleneck right now.

**No dedicated foot detector.** Off-the-shelf YOLO finds the shoe's position
correctly but has no "shoe" class, so it reports something else. Position
detection already works; only the classifier needs training.

**ROI is mount-specific.** Move the mirror and the values in `02_analyze.py`
need remeasuring.

---

## Next

- Collect many more stair and slope events, across users, sites and lighting.
- Fine-tune a foot/leg detector and pull coordinates instead of raw flow.
- Add a deceleration-aware fallback to the arrival estimate.
- Port the validated logic to Core ML and stream results to H-Medi over BLE.

Real-time inference has to run on the phone. Latency is a safety property here,
and a dropped network connection cannot be allowed to stop the estimate. Treat
the network as backup and retraining only.
