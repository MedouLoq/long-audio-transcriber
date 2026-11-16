"""User accounts for the transcriber (salted password hashes in a JSON file).

    python users.py add alice          # prompts for a password (12+ characters)
    python users.py remove alice
    python users.py list

The file defaults to data/users.json (override with USERS_FILE). Each user sees
only their own transcription history.
"""
import getpass
import json
import os
import re
import sys
import threading
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = Path(os.getenv("USERS_FILE", "").strip() or BASE_DIR / "data" / "users.json")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{2,40}$")
MIN_PASSWORD = 12
_lock = threading.Lock()


def load_users(path: Path = None) -> dict:
    path = path or USERS_FILE
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def save_users(users: dict, path: Path = None):
    path = path or USERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def set_password(name: str, password: str, path: Path = None):
    if not NAME_RE.match(name):
        raise ValueError("User name: 2-40 letters, digits, '.', '_' or '-'.")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    with _lock:
        users = load_users(path)
        users[name] = generate_password_hash(password)
        save_users(users, path)


def remove_user(name: str, path: Path = None) -> bool:
    with _lock:
        users = load_users(path)
        if users.pop(name, None) is None:
            return False
        save_users(users, path)
        return True


def verify(name: str, password: str, path: Path = None) -> bool:
    stored = load_users(path).get(name or "")
    return bool(stored) and check_password_hash(stored, password or "")


def main(argv):
    if len(argv) < 2 or argv[1] not in ("add", "remove", "list") or (argv[1] != "list" and len(argv) != 3):
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "list":
        for name in sorted(load_users()):
            print(name)
        return 0
    name = argv[2]
    if cmd == "remove":
        print("removed" if remove_user(name) else "no such user")
        return 0
    first = getpass.getpass(f"Password for {name}: ")
    if getpass.getpass("Repeat: ") != first:
        print("Passwords do not match.")
        return 1
    try:
        set_password(name, first)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"Saved {name} in {USERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
