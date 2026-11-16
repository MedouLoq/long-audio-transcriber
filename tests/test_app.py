import importlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_app(monkeypatch, tmp_path, **env):
    base = {"DEMO_MODE": "true", "DEMO_DELAY": "0", "JOBS_DIR": str(tmp_path / "jobs"),
            "USERS_FILE": str(tmp_path / "users.json"), "CHUNK_SECONDS": "20", "HOST": "127.0.0.1",
            "APP_PASSWORD": "", "LEGACY_OWNER": ""}
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, v)
    import users
    importlib.reload(users)
    if "app" in sys.modules:
        return importlib.reload(sys.modules["app"])
    import app
    return app


def basic(user, password):
    import base64
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture(scope="session")
def audio_file(tmp_path_factory):
    """50 s synthetic tone: splits into 3 parts at CHUNK_SECONDS=20."""
    import imageio_ffmpeg
    path = tmp_path_factory.mktemp("audio") / "tone.m4a"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=50", "-c:a", "aac", str(path)], check=True)
    return path


def wait_done(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        status = client.get(f"/jobs/{job_id}").get_json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_end_to_end_job_with_timestamp_shift(monkeypatch, tmp_path, audio_file):
    app = load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    resp = client.post("/jobs", data={"audio": (io.BytesIO(audio_file.read_bytes()), "meeting.m4a")},
                       content_type="multipart/form-data")
    assert resp.status_code == 202
    status = wait_done(client, resp.get_json()["job_id"])
    assert status["state"] == "done"
    assert status["chunks_total"] == 3 and status["chunks_done"] == 3
    assert 49 <= status["duration_seconds"] <= 51

    text = client.get(status["transcript_url"]).get_data(as_text=True)
    stamps = [line.split("]")[0] + "]" for line in text.splitlines() if line.startswith("[")]
    # parts start at 0 s, 20 s and 40 s of the recording
    assert stamps[0] == "[00:00:00]" and "[00:00:20]" in stamps and "[00:00:40]" in stamps
    assert any("Speaker numbers restart" in w for w in status["warnings"])
    # work files and the uploaded source are cleaned up
    assert sorted(p.name for p in (tmp_path / "jobs" / status["job_id"]).iterdir()) == ["status.json", "transcript.txt"]


def test_failed_part_is_marked_not_dropped(monkeypatch, tmp_path, audio_file):
    app = load_app(monkeypatch, tmp_path)
    real = app.transcribe_chunk

    def flaky(job_id, index, path):
        return ("", "empty: finish_reason=STOP") if index == 1 else real(job_id, index, path)

    monkeypatch.setattr(app, "transcribe_chunk", flaky)
    client = app.app.test_client()
    job = client.post("/jobs", data={"audio": (io.BytesIO(audio_file.read_bytes()), "a.m4a")},
                      content_type="multipart/form-data").get_json()["job_id"]
    status = wait_done(client, job)
    assert status["state"] == "done"
    text = client.get(f"/jobs/{job}/transcript.txt").get_data(as_text=True)
    assert "[00:00:20] [PART NOT TRANSCRIBED" in text
    assert any("missing from this transcript" in w for w in status["warnings"])


def test_all_parts_failing_fails_the_job(monkeypatch, tmp_path, audio_file):
    app = load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "transcribe_chunk", lambda *a: ("", "empty: x"))
    client = app.app.test_client()
    job = client.post("/jobs", data={"audio": (io.BytesIO(audio_file.read_bytes()), "a.m4a")},
                      content_type="multipart/form-data").get_json()["job_id"]
    status = wait_done(client, job)
    assert status["state"] == "failed" and "No part could be transcribed" in status["error"]


def test_non_audio_upload_fails_cleanly(monkeypatch, tmp_path):
    app = load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    job = client.post("/jobs", data={"audio": (io.BytesIO(b"not audio"), "x.mp3")},
                      content_type="multipart/form-data").get_json()["job_id"]
    status = wait_done(client, job)
    assert status["state"] == "failed" and "ffmpeg" in status["error"]


@pytest.mark.parametrize("text,offset,expected", [
    ("[00:05] hi", 0, "[00:00:05] hi"),
    ("[01:30] a\n[14:59] b", 900, "[00:16:30] a\n[00:29:59] b"),
    ("[1:02:03.5] c", 60, "[01:03:03] c"),
])
def test_shift_timestamps(monkeypatch, tmp_path, text, offset, expected):
    app = load_app(monkeypatch, tmp_path)
    assert app.shift_timestamps(text, offset) == expected


def test_password_protection(monkeypatch, tmp_path):
    app = load_app(monkeypatch, tmp_path, APP_PASSWORD="correct-horse-battery")
    client = app.app.test_client()
    assert client.get("/healthz").status_code == 401
    good, bad = basic("transcriber", "correct-horse-battery"), basic("transcriber", "wrong")
    assert client.get("/healthz", headers=bad).status_code == 401
    assert client.get("/healthz", headers=good).status_code == 200


def test_refuses_network_bind_without_strong_password(tmp_path):
    env = {**os.environ, "DEMO_MODE": "true", "HOST": "0.0.0.0", "APP_PASSWORD": "short", "JOBS_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-c", "import app"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "APP_PASSWORD" in proc.stderr


def test_job_ids_are_validated(monkeypatch, tmp_path):
    app = load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    for bad in ("..", "abc", "../../etc/passwd", "0" * 31):
        assert client.get(f"/jobs/{bad}").status_code == 404
        assert client.get(f"/jobs/{bad}/transcript.txt").status_code == 404
    assert client.get("/jobs/" + "0" * 32).status_code == 404


def test_interrupted_jobs_are_failed_on_restart(monkeypatch, tmp_path):
    job = "a" * 32
    (tmp_path / "jobs" / job).mkdir(parents=True)
    (tmp_path / "jobs" / job / "status.json").write_text(json.dumps({"job_id": job, "state": "running"}))
    app = load_app(monkeypatch, tmp_path)     # import runs fail_interrupted_jobs()
    status = app.read_status(job)
    assert status["state"] == "failed" and "restart" in status["error"]


def upload(client, audio_file, headers=None, **form):
    data = {"audio": (io.BytesIO(audio_file.read_bytes()), "Réunion chantier.m4a"), **form}
    resp = client.post("/jobs", data=data, content_type="multipart/form-data", headers=headers or {})
    assert resp.status_code == 202, resp.data
    return resp.get_json()["job_id"]


def test_history_is_private_per_user(monkeypatch, tmp_path, audio_file):
    app = load_app(monkeypatch, tmp_path)
    import users
    users.set_password("alice", "alice-password-123")
    users.set_password("bob", "bob-password-12345")
    client = app.app.test_client()
    alice, bob = basic("alice", "alice-password-123"), basic("bob", "bob-password-12345")
    assert client.get("/jobs").status_code == 401

    job = upload(client, audio_file, alice, language="French")
    status = client.get(f"/jobs/{job}", headers=alice).get_json()
    assert status["owner"] == "alice" and status["language"] == "French" and status["title"] == "Réunion chantier"
    assert [j["job_id"] for j in client.get("/jobs", headers=alice).get_json()["jobs"]] == [job]
    assert client.get("/jobs", headers=bob).get_json()["jobs"] == []
    # bob cannot read, download, rename or delete alice's job
    assert client.get(f"/jobs/{job}", headers=bob).status_code == 404
    assert client.get(f"/jobs/{job}/transcript.txt", headers=bob).status_code == 404
    assert client.get(f"/jobs/{job}/transcript.srt", headers=bob).status_code == 404
    assert client.patch(f"/jobs/{job}", json={"title": "x"}, headers=bob).status_code == 404
    assert client.delete(f"/jobs/{job}", headers=bob).status_code == 404
    assert client.get("/me", headers=alice).get_json()["user"] == "alice"


def test_rename_srt_and_delete(monkeypatch, tmp_path, audio_file):
    app = load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    job = upload(client, audio_file)
    wait_done(client, job)
    assert client.patch(f"/jobs/{job}", json={"title": "Site meeting 12 March"}).get_json()["title"] == "Site meeting 12 March"
    assert client.patch(f"/jobs/{job}", json={"title": "  "}).status_code == 400
    srt = client.get(f"/jobs/{job}/transcript.srt")
    assert "Site_meeting_12_March.srt" in srt.headers["Content-Disposition"]
    body = srt.get_data(as_text=True)
    assert body.startswith("1\n00:00:00,000 --> 00:00:") and "\n2\n" in body
    assert client.delete(f"/jobs/{job}").status_code == 204
    assert client.get("/jobs").get_json()["jobs"] == []


def test_to_srt_timing(monkeypatch, tmp_path):
    app = load_app(monkeypatch, tmp_path)
    srt = app.to_srt("[00:00:05] Speaker 1: a\n[00:01:00] Speaker 2: b\n\n[00:01:10] [PART NOT TRANSCRIBED]")
    cues = srt.strip().split("\n\n")
    assert cues[0] == "1\n00:00:05,000 --> 00:00:35,000\nSpeaker 1: a"     # capped at 30 s
    assert cues[1] == "2\n00:01:00,000 --> 00:01:10,000\nSpeaker 2: b"     # ends at next cue
    assert cues[2].startswith("3\n00:01:10,000 --> 00:01:15,000")


def test_jobs_from_before_accounts_belong_to_legacy_owner(monkeypatch, tmp_path):
    job = "b" * 32
    (tmp_path / "jobs" / job).mkdir(parents=True)
    (tmp_path / "jobs" / job / "status.json").write_text(json.dumps({"job_id": job, "state": "done", "created_at": 1}))
    (tmp_path / "jobs" / job / "transcript.txt").write_text("[00:00:01] Speaker 1: old\n")
    app = load_app(monkeypatch, tmp_path, LEGACY_OWNER="alice")
    import users
    users.set_password("alice", "alice-password-123")
    users.set_password("bob", "bob-password-12345")
    client = app.app.test_client()
    assert len(client.get("/jobs", headers=basic("alice", "alice-password-123")).get_json()["jobs"]) == 1
    assert client.get("/jobs", headers=basic("bob", "bob-password-12345")).get_json()["jobs"] == []


def test_users_cli_rules(tmp_path):
    import users
    f = tmp_path / "u.json"
    with pytest.raises(ValueError):
        users.set_password("alice", "short", f)
    with pytest.raises(ValueError):
        users.set_password("bad name!", "long-enough-password", f)
    users.set_password("alice", "long-enough-password", f)
    stored = json.loads(f.read_text())["alice"]
    assert "long-enough-password" not in stored and users.verify("alice", "long-enough-password", f)
    assert not users.verify("alice", "wrong", f) and users.remove_user("alice", f) and not users.verify("alice", "long-enough-password", f)
