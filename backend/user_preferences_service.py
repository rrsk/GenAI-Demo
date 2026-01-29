"""
User Preferences Service - Learning system storage
Stores preference responses from dynamic UI components with atomic JSON writes.
"""
import json
import os
import re
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from threading import Lock

from .config import USER_DATA_DIR


class UserPreferencesService:
    """Store and load user preferences with atomic writes and in-memory cache."""

    QUESTION_COOLDOWN = timedelta(minutes=2)
    _cache: Dict[str, dict] = {}
    _cache_lock = Lock()

    def __init__(self):
        self.prefs_dir = USER_DATA_DIR / "preferences"
        self.prefs_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, user_id: str) -> Path:
        if not re.match(r"^[A-Za-z0-9_-]+$", user_id) or len(user_id) > 64:
            raise ValueError(f"Invalid user_id format: {user_id}")
        return self.prefs_dir / f"{user_id}.json"

    def _create_default_preferences(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "preferences": {},
            "questions_asked": [],
            "last_question_time": None,
        }

    def _atomic_write(self, path: Path, data: dict) -> None:
        fd, temp_path = tempfile.mkstemp(dir=str(self.prefs_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def load_preferences(self, user_id: str) -> dict:
        with self._cache_lock:
            if user_id in self._cache:
                return dict(self._cache[user_id])

        file_path = self._get_file_path(user_id)
        if not file_path.exists():
            return self._create_default_preferences(user_id)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            with self._cache_lock:
                self._cache[user_id] = data
            return dict(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[PrefsService] Error loading {user_id}: {e}")
            return self._create_default_preferences(user_id)

    def save_preference(
        self,
        user_id: str,
        question_id: str,
        category: str,
        selected_options: List[str],
        skipped: bool = False,
    ) -> None:
        if not re.match(r"^[a-z0-9_]+$", question_id) or len(question_id) > 50:
            raise ValueError("Invalid question_id")
        if category not in ("dietary", "schedule", "goals", "feedback"):
            raise ValueError("Invalid category")

        prefs = self.load_preferences(user_id)
        if "preferences" not in prefs:
            prefs["preferences"] = {}
        if category not in prefs["preferences"]:
            prefs["preferences"][category] = {}
        prefs["preferences"][category][question_id] = selected_options if not skipped else []
        prefs["updated_at"] = datetime.utcnow().isoformat()

        path = self._get_file_path(user_id)
        self._atomic_write(path, prefs)
        with self._cache_lock:
            self._cache[user_id] = prefs

    def can_ask_question(self, user_id: str) -> bool:
        prefs = self.load_preferences(user_id)
        last_time = prefs.get("last_question_time")
        if not last_time:
            return True
        try:
            last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return True
        return datetime.utcnow() - last_dt.replace(tzinfo=None) > self.QUESTION_COOLDOWN

    def record_question_asked(self, user_id: str, question_id: str) -> None:
        prefs = self.load_preferences(user_id)
        prefs["last_question_time"] = datetime.utcnow().isoformat()
        if "questions_asked" not in prefs:
            prefs["questions_asked"] = []
        if question_id not in prefs["questions_asked"]:
            prefs["questions_asked"].append(question_id)
        prefs["updated_at"] = datetime.utcnow().isoformat()
        path = self._get_file_path(user_id)
        self._atomic_write(path, prefs)
        with self._cache_lock:
            self._cache[user_id] = prefs

    def get_context_string(self, user_id: str) -> str:
        prefs = self.load_preferences(user_id)
        pref_data = prefs.get("preferences") or {}
        if not pref_data:
            return "No preferences recorded yet."
        lines = ["User Preferences:"]
        for category, questions in pref_data.items():
            lines.append(f"\n## {category.title()}")
            for q_id, values in questions.items():
                if isinstance(values, list):
                    lines.append(f"- {q_id}: {', '.join(values)}")
                else:
                    lines.append(f"- {q_id}: {values}")
        return "\n".join(lines)


_prefs_service: Optional[UserPreferencesService] = None


def get_preferences_service() -> UserPreferencesService:
    global _prefs_service
    if _prefs_service is None:
        _prefs_service = UserPreferencesService()
    return _prefs_service
