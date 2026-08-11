"""
Turn the session audio into timestamped motion labels.

Run this inside a session folder:
    python3 01_make_labels.py

Reads  : audio.m4a, audio_start.txt
Writes : labels.csv  (hostTime, label, conf, raw)

Whisper gives us word-level timestamps relative to the start of the audio
file. Adding audio_start puts them on the same host clock as the depth and
IMU logs, so labels line up with everything else without any extra sync step.
"""

import os
import re
import csv
import shutil
from collections import Counter

import imageio_ffmpeg

# Whisper shells out to `ffmpeg`, so make sure the bundled binary is on PATH
# under that exact name. Saves anyone from having to install it separately.
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(FFMPEG) + os.pathsep + os.environ.get("PATH", "")
_alias = os.path.join(os.path.dirname(FFMPEG), "ffmpeg")
if not os.path.exists(_alias):
    try:
        os.symlink(FFMPEG, _alias)
    except OSError:
        shutil.copy(FFMPEG, _alias)

import whisper  # noqa: E402  (import after PATH is patched)

HERE = os.path.dirname(os.path.abspath(__file__))

CONF_MIN = 0.4   # below this a hit goes to the "suspect" pile and is dropped
MERGE = 1.2      # seconds; repeats of the same label inside this window collapse

# Two-word labels are checked first and consume both words, otherwise "stair"
# would match on its own and swallow the direction.
MULTI = [
    (["계단위", "계단올라", "계단워", "계단 위"], "stair_up"),
    (["계단아래", "계단내려", "계단알", "계단 아래"], "stair_down"),
]

# Each entry lists the intended word plus mishearings we've actually seen in
# transcripts. Add to these rather than loosening the matching.
SINGLE = [
    (["시작", "스타트"], "start"),
    (["끝", "종료"], "end"),
    (["오르막", "물음악", "오르마"], "slope_up"),
    (["내리막", "내리마"], "slope_down"),
    (["바닥", "바다", "바닦"], "floor"),
    (["정지", "멈춤", "스톱"], "stop"),
    (["회전", "돌아서", "턴"], "turn"),
    (["문", "도어"], "door"),
]


def strip_punct(s):
    """Transcripts come back with commas and periods glued to words."""
    return re.sub(r"[,\.\?\!\s]", "", s)


def transcribe(audio_path):
    print("loading whisper large-v3 (first run downloads ~3 GB)")
    model = whisper.load_model("large-v3")
    print("transcribing")
    return model.transcribe(
        audio_path,
        language="ko",
        word_timestamps=True,
        # Leave this off. With it on, Whisper conditions each window on its own
        # previous output and gets stuck in a loop when the same short command
        # repeats — whole stair sections came back transcribed as "바닥".
        condition_on_previous_text=False,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4),
        compression_ratio_threshold=2.0,
        no_speech_threshold=0.5,
        initial_prompt="시작 끝 바닥 오르막 내리막 계단 위 계단 아래 정지 회전 문",
    )


def flatten_words(result):
    words = []
    for seg in result["segments"]:
        for wd in seg.get("words") or []:
            w = strip_punct(wd["word"])
            if w:
                words.append({"w": w, "t": wd["start"], "p": wd.get("probability", 1.0)})
    return words


def match_labels(words, audio_start):
    hits = []
    i, n = 0, len(words)
    while i < n:
        found, consume = None, 1

        if i + 1 < n:
            pair = words[i]["w"] + words[i + 1]["w"]
            for keys, label in MULTI:
                if any(k in pair for k in keys):
                    found = {"label": label, "raw": pair,
                             "conf": (words[i]["p"] + words[i + 1]["p"]) / 2}
                    consume = 2
                    break

        if not found:
            # Sometimes the direction ends up inside a single token.
            word = words[i]["w"]
            for keys, label in MULTI:
                if any(k in word for k in keys):
                    found = {"label": label, "raw": word, "conf": words[i]["p"]}
                    break

        if not found:
            word = words[i]["w"]
            for keys, label in SINGLE:
                if any(k in word for k in keys):
                    found = {"label": label, "raw": word, "conf": words[i]["p"]}
                    break

        if found:
            found["host"] = audio_start + words[i]["t"]
            found["low"] = found["conf"] < CONF_MIN
            hits.append(found)
        i += consume
    return hits


def dedupe(hits):
    """Collapse repeats of the same label only. Two different labels spoken
    back to back are both real and must survive."""
    out = []
    for h in hits:
        prior = next((x for x in out
                      if x["label"] == h["label"] and abs(h["host"] - x["host"]) < MERGE), None)
        if prior:
            if h["conf"] > prior["conf"]:
                prior.update(h)
        else:
            out.append(h)
    out.sort(key=lambda x: x["host"])
    return out


def main():
    with open(os.path.join(HERE, "audio_start.txt")) as f:
        audio_start = float(f.read().strip())
    print(f"audio start hostTime: {audio_start:.3f}")

    result = transcribe(os.path.join(HERE, "audio.m4a"))
    labels = dedupe(match_labels(flatten_words(result), audio_start))

    print("\n--- transcript ---")
    for seg in result["segments"]:
        print(f"[{seg['start']:7.2f}s] {seg['text'].strip()}")

    print(f"\n--- {len(labels)} labels ---")
    print(f"{'time(s)':>9} {'label':12} {'conf':>6}  heard")
    for L in labels:
        flag = "suspect" if L["low"] else "ok"
        print(f"{L['host'] - audio_start:9.1f} {L['label']:12} {L['conf']:6.2f} {flag:8} \"{L['raw']}\"")

    print("\ncounts:", dict(Counter(L["label"] for L in labels)))

    keep = [L for L in labels if not L["low"]]
    with open(os.path.join(HERE, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hostTime", "label", "conf", "raw"])
        for L in keep:
            w.writerow([f"{L['host']:.6f}", L["label"], f"{L['conf']:.3f}", L["raw"]])

    print(f"\nwrote labels.csv ({len(keep)} labels)")
    print("Skim the transcript above — deleting a bad row by hand is fine.")


if __name__ == "__main__":
    main()
