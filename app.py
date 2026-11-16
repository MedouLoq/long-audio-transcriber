"""Long-audio transcription service: Flask + ffmpeg + Cloud Storage + Gemini on Vertex AI.

Configure with environment variables (see .env.example), then:
    pip install -r requirements.txt
    python app.py                      # http://127.0.0.1:5000
    DEMO_MODE=true python app.py       # no Google Cloud needed: model calls are simulated

Flow: upload -> background job -> ffmpeg normalises and splits the audio ->
each chunk goes to GCS and is transcribed by Gemini -> chunk timestamps are
shifted to recording time -> transcript is stitched and served.
"""
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, request, send_from_directory
from google.genai import errors, types
from werkzeug.utils import secure_filename

import users

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcriber")
for _noisy in ("google_genai", "google_genai.models", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ============================ SETTINGS ============================
# All settings come from environment variables (or a .env file): see .env.example.
BASE_DIR = Path(__file__).resolve().parent


def _env(name, default=""):
    return os.getenv(name, default).strip()


KEY_FILE = _env("GOOGLE_APPLICATION_CREDENTIALS")   # service-account JSON; empty = Application Default Credentials
GCP_PROJECT = _env("GCP_PROJECT")                    # optional when the key file names its project
BUCKET_NAME = _env("BUCKET_NAME")                    # empty = "<project>-transcriber-audio"
BUCKET_LOCATION = _env("BUCKET_LOCATION", "us-central1")
LOCATION = _env("VERTEX_LOCATION", "global")         # use "us-central1" if "global" gives a 404
MODEL = _env("MODEL", "gemini-3.8-flash")
CHUNK_SECONDS = int(_env("CHUNK_SECONDS", "900"))    # length of each part sent to the model
PARALLEL_CHUNKS = int(_env("PARALLEL_CHUNKS", "3"))  # parts transcribed at once
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "2048"))
LANGUAGE_HINT = _env("LANGUAGE_HINT")                # e.g. "Arabic and French"
THINKING_LEVEL = _env("THINKING_LEVEL", "LOW")       # MINIMAL / LOW / MEDIUM / HIGH, or "" for the default
FALLBACK_MODEL = _env("FALLBACK_MODEL", "gemini-3.5-flash")  # tried once when MODEL keeps answering empty; "" = off
HOST = _env("HOST", "127.0.0.1")
PORT = int(_env("PORT", "5000"))
APP_USER = _env("APP_USER", "transcriber")      # optional single shared login (legacy)
APP_PASSWORD = _env("APP_PASSWORD")
LEGACY_OWNER = _env("LEGACY_OWNER") or APP_USER  # owner of jobs created before per-user history
JOBS_DIR = Path(_env("JOBS_DIR") or BASE_DIR / "data" / "jobs")
DEMO_MODE = _env("DEMO_MODE").lower() in ("1", "true", "yes", "on")
# ==================================================================


def auth_enabled():
    return bool(users.load_users()) or bool(APP_PASSWORD)


if HOST not in ("127.0.0.1", "localhost", "::1") and not users.load_users() and len(APP_PASSWORD) < 12:
    raise SystemExit("Create a user (python users.py add <name>) or set APP_PASSWORD (12+ characters) "
                     "before binding to a network interface: without it anyone who finds the port can "
                     "read transcripts and spend your Google Cloud budget. For local use keep HOST=127.0.0.1.")

if DEMO_MODE:
    CREDENTIALS, PROJECT_ID = None, "demo"
else:
    import google.auth
    from google.oauth2 import service_account

    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    if KEY_FILE:
        if not Path(KEY_FILE).is_file():
            raise SystemExit(f"Service-account key not found: {KEY_FILE} (GOOGLE_APPLICATION_CREDENTIALS)")
        CREDENTIALS = service_account.Credentials.from_service_account_file(KEY_FILE, scopes=_SCOPES)
        PROJECT_ID = GCP_PROJECT or CREDENTIALS.project_id
    else:
        CREDENTIALS, _default_project = google.auth.default(scopes=_SCOPES)
        PROJECT_ID = GCP_PROJECT or _default_project
    if not PROJECT_ID:
        raise SystemExit("Could not determine the Google Cloud project: set GCP_PROJECT.")
BUCKET_NAME = BUCKET_NAME or f"{PROJECT_ID}-transcriber-audio"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run: pip install imageio-ffmpeg")


FFMPEG = _find_ffmpeg()

EMPTY_RETRIES = 2  # the model sometimes answers 200 OK with no text; ask again
RETRYABLE = {408, 429, 500, 502, 503, 504}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TS_RE = re.compile(r"\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.\d+)?\]")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_job_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
_chunk_pool = ThreadPoolExecutor(max_workers=PARALLEL_CHUNKS, thread_name_prefix="chunk")
_status_lock = threading.Lock()
_clients = {}
_clients_lock = threading.Lock()


@app.before_request
def require_login():
    """HTTP basic auth against users.json (hashed) or the legacy APP_USER/APP_PASSWORD pair."""
    if not auth_enabled():
        g.user = "local"           # no accounts configured: single local user
        return None
    auth = request.authorization
    if auth and auth.type == "basic":
        name, password = auth.username or "", auth.password or ""
        if users.verify(name, password):
            g.user = name
            return None
        if (APP_PASSWORD and hmac.compare_digest(name.encode(), APP_USER.encode())
                and hmac.compare_digest(password.encode(), APP_PASSWORD.encode())):
            g.user = APP_USER
            return None
    return Response("Login required.", 401, {"WWW-Authenticate": 'Basic realm="Transcriber", charset="UTF-8"'})


# --- Google clients (created lazily, shared across threads) -----------------

def get_genai():
    from google import genai

    with _clients_lock:
        if "genai" not in _clients:
            _clients["genai"] = genai.Client(
                vertexai=True,
                credentials=CREDENTIALS,
                project=PROJECT_ID,
                location=LOCATION,
                http_options=types.HttpOptions(timeout=600_000),  # milliseconds
            )
        return _clients["genai"]


def get_bucket():
    """Returns the bucket, creating it (with 1-day auto-delete) on first use."""
    from google.api_core import exceptions as gexc
    from google.cloud import storage

    with _clients_lock:
        if "bucket" not in _clients:
            client = storage.Client(project=PROJECT_ID, credentials=CREDENTIALS)
            bucket = client.bucket(BUCKET_NAME)
            try:
                exists = bucket.exists()
            except gexc.Forbidden:
                exists = True  # key may write objects without being allowed to inspect the bucket
            if not exists:
                bucket.iam_configuration.uniform_bucket_level_access_enabled = True
                bucket.add_lifecycle_delete_rule(age=1)  # backstop for leftovers
                try:
                    client.create_bucket(bucket, location=BUCKET_LOCATION)
                    log.info("Created bucket gs://%s", BUCKET_NAME)
                except gexc.Forbidden:
                    raise RuntimeError(
                        f"Bucket gs://{BUCKET_NAME} does not exist and this key may not create it. "
                        f"Create it in the Cloud console and give {CREDENTIALS.service_account_email} "
                        "the 'Storage Object User' role on it, or set BUCKET_NAME to a bucket it can use.")
            _clients["bucket"] = bucket
        return _clients["bucket"]


# --- Job state on disk ------------------------------------------------------

def job_dir(job_id):
    if not JOB_ID_RE.match(job_id):
        abort(404)
    return JOBS_DIR / job_id


def read_status(job_id):
    path = JOBS_DIR / job_id / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def owner_of(status):
    return status.get("owner") or LEGACY_OWNER


def own_status(job_id):
    """Status of a job that belongs to the signed-in user, else 404 (no leaking of other users' ids)."""
    job_dir(job_id)
    status = read_status(job_id)
    if status is None or owner_of(status) != g.user and auth_enabled():
        abort(404)
    return status


def write_status(job_id, **changes):
    with _status_lock:
        path = JOBS_DIR / job_id / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {"job_id": job_id}
        status.update(changes, updated_at=time.time())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        tmp.replace(path)
        return status


def fail_interrupted_jobs():
    """Jobs that were mid-flight when the process stopped can never finish."""
    for path in JOBS_DIR.glob("*/status.json"):
        status = json.loads(path.read_text())
        if status.get("state") in ("queued", "running"):
            write_status(status["job_id"], state="failed",
                         error="Interrupted by a service restart. Upload again.")
            shutil.rmtree(path.parent / "work", ignore_errors=True)


# --- Audio preparation ------------------------------------------------------

def split_audio(source, work_dir):
    """Normalise to mono 16 kHz MP3 and split. Returns ([(path, offset_seconds)], duration)."""
    listing = work_dir / "segments.csv"
    proc = subprocess.run(
        [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "48k",
         "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
         "-segment_list", str(listing), "-segment_list_type", "csv",
         str(work_dir / "chunk_%04d.mp3")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read this file: {proc.stderr.strip()[-300:]}")
    chunks, duration = [], 0.0
    if listing.exists():
        for line in listing.read_text().splitlines():  # name,start,end in recording time
            name, start, end = line.rsplit(",", 2)
            chunks.append((work_dir / name, float(start)))
            duration = float(end)
    if not chunks:
        raise RuntimeError("No audio stream found in the uploaded file.")
    return chunks, duration


# --- Transcription ----------------------------------------------------------

def build_prompt(language_hint=""):
    hint = language_hint or LANGUAGE_HINT
    language = (f"The recording is mainly in: {hint}. " if hint else "")
    return (
        "Transcribe this audio clip verbatim. " + language +
        "Write every utterance in the language and script it was spoken in; do not "
        "translate, summarise, or correct the speakers. "
        "Start a new line at each speaker change, formatted as "
        "'[MM:SS] Speaker N: text', where MM:SS is the time from the start of THIS "
        "clip. Also add a timestamp at least every 30 seconds within a long turn. "
        "Number speakers in order of first appearance. Use [inaudible] for speech you "
        "cannot make out and [silence] or [music] for long non-speech stretches. "
        "Output only the transcript, with no introduction or closing remarks."
    )


# Transcription must reproduce what was said, so the content filters are switched off;
# left on, they silently return an empty answer for a whole part.
SAFETY_OFF = [
    types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.OFF)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]
_thinking_rejected = set()  # models that refused THINKING_LEVEL with a 400


def build_config(model):
    thinking = None
    if THINKING_LEVEL and model not in _thinking_rejected:
        thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    return types.GenerateContentConfig(safety_settings=SAFETY_OFF, thinking_config=thinking)


def generate_with_retry(gcs_uri, model, prompt):
    part = types.Part.from_uri(file_uri=gcs_uri, mime_type="audio/mpeg")
    delay = 5
    for attempt in range(1, 6):
        config = build_config(model)
        try:
            return get_genai().models.generate_content(
                model=model, contents=[part, prompt], config=config)
        except errors.APIError as exc:
            if exc.code == 400 and config.thinking_config is not None:  # this request sent it
                log.warning("%s rejected the request (%s); retrying without THINKING_LEVEL",
                            model, exc.message)
                _thinking_rejected.add(model)
                continue
            if exc.code not in RETRYABLE or attempt == 5:
                raise
            log.warning("Gemini %s on attempt %d, retrying in %ds", exc.code, attempt, delay)
            time.sleep(delay)
            delay *= 2


def describe_response(response):
    """One line of diagnostics for a response that carried no transcript."""
    candidate = response.candidates[0] if response.candidates else None
    parts = (candidate.content.parts if candidate and candidate.content else None) or []
    usage = response.usage_metadata
    return (f"finish_reason={candidate.finish_reason if candidate else None}, "
            f"parts={len(parts)}, thought_parts={sum(1 for p in parts if getattr(p, 'thought', False))}, "
            f"prompt_tokens={getattr(usage, 'prompt_token_count', None)}, "
            f"output_tokens={getattr(usage, 'candidates_token_count', None)}, "
            f"thinking_tokens={getattr(usage, 'thoughts_token_count', None)}, "
            f"prompt_feedback={response.prompt_feedback}")


def transcribe_chunk(job_id, index, path):
    """Returns (text, problem). problem is None, "truncated", or "empty: <diagnostics>"."""
    if DEMO_MODE:
        from demo import demo_transcribe
        return demo_transcribe(index, path, FFMPEG, delay=float(_env("DEMO_DELAY", "1.5")))
    blob = get_bucket().blob(f"transcriptions/{job_id}/{path.name}")
    blob.upload_from_filename(str(path), content_type="audio/mpeg")
    try:
        detail = ""
        uri = f"gs://{BUCKET_NAME}/{blob.name}"
        prompt = build_prompt((read_status(job_id) or {}).get("language", ""))
        models = [MODEL] * EMPTY_RETRIES + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])
        for attempt, model in enumerate(models, 1):
            try:
                response = generate_with_retry(uri, model, prompt)
            except errors.APIError as exc:
                if model == MODEL:
                    raise
                detail = f"fallback {model} failed: {exc.code} {exc.message}"
                log.warning("Part %d: %s", index + 1, detail)
                break
            text = (response.text or "").strip()
            if text:
                if model != MODEL:
                    log.info("Part %d: transcribed by fallback model %s", index + 1, model)
                reason = response.candidates[0].finish_reason if response.candidates else None
                return text, ("truncated" if reason == types.FinishReason.MAX_TOKENS else None)
            detail = describe_response(response)
            log.warning("Part %d: empty response from %s, attempt %d/%d (%s)",
                        index + 1, model, attempt, len(models), detail)
        return "", f"empty: {detail}"
    finally:
        try:
            blob.delete()
        except Exception as exc:  # bucket lifecycle rule is the backstop
            log.warning("Could not delete %s: %s", blob.name, exc)


def shift_timestamps(text, offset):
    def repl(match):
        hours, minutes, seconds = (int(g or 0) for g in match.groups())
        total = int(round(hours * 3600 + minutes * 60 + seconds + offset))
        return f"[{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}]"
    return TS_RE.sub(repl, text)


def run_job(job_id, source):
    work = JOBS_DIR / job_id / "work"
    try:
        write_status(job_id, state="running", stage="preparing audio")
        chunks, duration = split_audio(source, work)
        source.unlink(missing_ok=True)
        write_status(job_id, stage="transcribing", duration_seconds=round(duration),
                     chunks_total=len(chunks), chunks_done=0)

        done = 0
        progress_lock = threading.Lock()

        def work_one(item):
            nonlocal done
            index, (path, _offset) = item
            try:
                result = transcribe_chunk(job_id, index, path)
            except Exception as exc:  # one bad part must not throw away the others
                log.exception("Part %d failed", index + 1)
                result = ("", f"error: {exc}")
            with progress_lock:  # count and write together so progress never goes backwards
                done += 1
                write_status(job_id, chunks_done=done)
            return result

        results = list(_chunk_pool.map(work_one, enumerate(chunks)))
        if all(problem and not text for text, problem in results):
            raise RuntimeError(f"No part could be transcribed. First problem: {results[0][1]}")

        warnings, parts = [], []
        for (path, offset), (text, problem) in zip(chunks, results):
            start = shift_timestamps("[00:00]", offset)
            if problem == "truncated":
                warnings.append(f"The part starting at {start} hit the model's output limit "
                                "and is cut short. Lower CHUNK_SECONDS and run it again.")
            elif problem:
                minutes = CHUNK_SECONDS // 60
                warnings.append(f"The {minutes}-minute part starting at {start} is missing "
                                f"from this transcript ({problem.split(':')[0]}). Upload the file again to retry.")
                text = f"{start} [PART NOT TRANSCRIBED: about {minutes} minutes missing here]"
            parts.append(shift_timestamps(text, offset) if not problem or problem == "truncated" else text)
        if len(chunks) > 1:
            warnings.append(f"Speaker numbers restart every {CHUNK_SECONDS // 60} minutes: "
                            "'Speaker 1' in one part may be a different person in the next.")

        (JOBS_DIR / job_id / "transcript.txt").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        write_status(job_id, state="done", stage="done", warnings=warnings)
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        write_status(job_id, state="failed", error=str(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        source.unlink(missing_ok=True)


# --- HTTP -------------------------------------------------------------------

@app.post("/jobs")
def create_job():
    upload = request.files.get("audio")
    if upload is None or upload.filename == "":
        return jsonify(error="Attach a file in the 'audio' form field."), 400

    job_id = uuid.uuid4().hex
    work = JOBS_DIR / job_id / "work"
    work.mkdir(parents=True)
    name = secure_filename(upload.filename) or "upload"
    source = work / f"source{Path(name).suffix[:10]}"
    upload.save(source)  # streamed to disk, not held in memory

    language = (request.form.get("language") or "").strip()[:60]
    # Display title keeps the original name (accents, spaces); it is only ever rendered as text
    original = re.split(r"[\\/]", upload.filename)[-1]
    title = (original.rsplit(".", 1)[0] if "." in original else original).strip()[:120] or Path(name).stem
    write_status(job_id, state="queued", filename=name, title=title, owner=g.user,
                 language=language, size_bytes=source.stat().st_size, created_at=time.time())
    _job_pool.submit(run_job, job_id, source)
    return jsonify(job_id=job_id, status_url=f"/jobs/{job_id}"), 202


def public_status(status):
    status = dict(status)
    status.setdefault("title", Path(status.get("filename", "recording")).stem)
    if status["state"] == "done":
        status["transcript_url"] = f"/jobs/{status['job_id']}/transcript.txt"
    return status


@app.get("/jobs")
def list_jobs():
    """The signed-in user's history, newest first."""
    jobs = []
    for path in JOBS_DIR.glob("*/status.json"):
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if auth_enabled() and owner_of(status) != g.user:
            continue
        jobs.append(public_status(status))
    jobs.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return jsonify(jobs=jobs)


@app.get("/me")
def me():
    return jsonify(user=g.user, auth=auth_enabled(), demo=DEMO_MODE, chunk_minutes=CHUNK_SECONDS // 60,
                   max_upload_mb=MAX_UPLOAD_MB)


@app.get("/jobs/<job_id>")
def get_job(job_id):
    return jsonify(public_status(own_status(job_id)))


@app.patch("/jobs/<job_id>")
def rename_job(job_id):
    own_status(job_id)
    title = ((request.get_json(silent=True) or {}).get("title") or "").strip()[:120]
    if not title:
        return jsonify(error="Title cannot be empty."), 400
    return jsonify(public_status(write_status(job_id, title=title)))


def _transcript_text(job_id):
    own_status(job_id)
    path = JOBS_DIR / job_id / "transcript.txt"
    if not path.exists():
        abort(404)
    return path.read_text(encoding="utf-8")


def _download_name(job_id, ext):
    title = (read_status(job_id) or {}).get("title") or f"transcript-{job_id[:8]}"
    return secure_filename(title) or f"transcript-{job_id[:8]}"


@app.get("/jobs/<job_id>/transcript.txt")
def get_transcript(job_id):
    text = _transcript_text(job_id)
    headers = {}
    if request.args.get("download"):
        headers["Content-Disposition"] = f'attachment; filename="{_download_name(job_id, "txt")}.txt"'
    return Response(text, mimetype="text/plain; charset=utf-8", headers=headers)


LINE_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")


def to_srt(text, last_cue_seconds=5, max_cue_seconds=30):
    """Transcript lines '[HH:MM:SS] Speaker N: text' → SubRip subtitles."""
    cues = []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if m and m[4]:
            cues.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), m[4]))

    def ts(sec):
        return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d},000"

    out = []
    for i, (start, body) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else start + last_cue_seconds
        end = max(start + 1, min(nxt, start + max_cue_seconds))
        out.append(f"{i + 1}\n{ts(start)} --> {ts(end)}\n{body}\n")
    return "\n".join(out)


@app.get("/jobs/<job_id>/transcript.srt")
def get_srt(job_id):
    srt = to_srt(_transcript_text(job_id))
    return Response(srt, mimetype="application/x-subrip; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{_download_name(job_id, "srt")}.srt"'})


@app.delete("/jobs/<job_id>")
def delete_job(job_id):
    status = own_status(job_id)
    if status["state"] in ("queued", "running"):
        return jsonify(error="Job is still running."), 409
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    return "", 204


@app.errorhandler(413)
def too_large(_exc):
    return jsonify(error=f"File is larger than the {MAX_UPLOAD_MB} MB limit."), 413


@app.get("/healthz")
def healthz():
    return jsonify(ok=True, model="demo" if DEMO_MODE else MODEL, location=LOCATION)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")


fail_interrupted_jobs()

if __name__ == "__main__":
    from waitress import serve
    log.info("Project %s | bucket gs://%s | model %s | http://%s:%s%s",
             PROJECT_ID, BUCKET_NAME, MODEL, HOST, PORT, " | DEMO MODE" if DEMO_MODE else "")
    serve(app, host=HOST, port=PORT, threads=8, max_request_body_size=MAX_UPLOAD_MB * 1024 * 1024)