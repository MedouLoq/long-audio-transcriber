"""DEMO_MODE: a stand-in for the Gemini call.

Everything else in the pipeline is real: ffmpeg normalises and splits the
upload, parts run in parallel, timestamps are shifted to recording time and
the transcript is stitched. Only the text below is invented: a fictional
meeting, with timestamps relative to each part, exactly as the model would
return them.
"""
import re
import subprocess
import time

LINES = [
    "Bonjour à tous, on commence la réunion de suivi du projet.",
    "Merci. Je propose qu'on passe d'abord sur le planning de la semaine.",
    "D'accord. Les relevés de terrain sont terminés à quatre-vingts pour cent.",
    "Est-ce qu'on a reçu les validations pour la deuxième zone ?",
    "Pas encore, la réponse est attendue pour jeudi.",
    "Very good, then we can schedule the review for Friday morning.",
    "Ok, je note : revue vendredi matin, et envoi du rapport lundi.",
    "Il reste la question du budget pour le matériel.",
    "[inaudible] … on en reparle après la pause.",
    "Parfait. Quelqu'un a d'autres points à ajouter ?",
]


def _duration_seconds(path, ffmpeg) -> float:
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 60.0


def demo_transcribe(index: int, path, ffmpeg: str, delay: float = 1.5):
    """Returns (text, problem) like transcribe_chunk()."""
    time.sleep(delay)  # pretend to wait for the model
    duration = _duration_seconds(path, ffmpeg)
    rows, t, line, speaker = [], 0, index * 3, 1
    while t < duration - 5:
        rows.append(f"[{t // 60:02d}:{t % 60:02d}] Speaker {speaker}: {LINES[line % len(LINES)]}")
        line += 1
        speaker = speaker % 3 + 1
        t += 25 + (line * 7) % 20
    return "\n".join(rows), None
