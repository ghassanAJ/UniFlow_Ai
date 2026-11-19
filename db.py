"""SQLite persistence for students, courses, and enrollments."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "./uniflow.db"))
LEGACY_JSON = Path(os.getenv("ENROLLMENTS_PATH", "./enrollments.json"))


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS enrollments (
                student_id TEXT NOT NULL,
                course_id TEXT NOT NULL,
                PRIMARY KEY (student_id, course_id),
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
            );
            """
        )
    _migrate_legacy_json()
    _seed_if_empty()


def _migrate_legacy_json() -> None:
    if not LEGACY_JSON.exists():
        return
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
        if count:
            return
        data = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
        for student_id, courses in data.items():
            conn.execute(
                "INSERT OR IGNORE INTO students(student_id, name) VALUES (?, ?)",
                (student_id, student_id),
            )
            for course_id in courses:
                conn.execute(
                    "INSERT OR IGNORE INTO courses(course_id, title) VALUES (?, ?)",
                    (course_id, course_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO enrollments(student_id, course_id) VALUES (?, ?)",
                    (student_id, course_id),
                )


def _seed_if_empty() -> None:
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
        if count:
            return
        seed = {
            "s1": ("Student 1", ["CS101", "MATH201"]),
            "s2": ("Student 2", ["CS101", "PHYS110"]),
        }
        for sid, (name, courses) in seed.items():
            conn.execute(
                "INSERT INTO students(student_id, name) VALUES (?, ?)",
                (sid, name),
            )
            for cid in courses:
                conn.execute(
                    "INSERT OR IGNORE INTO courses(course_id, title) VALUES (?, ?)",
                    (cid, cid),
                )
                conn.execute(
                    "INSERT INTO enrollments(student_id, course_id) VALUES (?, ?)",
                    (sid, cid),
                )


def list_courses() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT course_id, title FROM courses ORDER BY course_id"
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_course(course_id: str, title: str = "") -> dict:
    course_id = course_id.strip()
    title = (title or course_id).strip()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO courses(course_id, title) VALUES (?, ?)
            ON CONFLICT(course_id) DO UPDATE SET title = excluded.title
            """,
            (course_id, title),
        )
    return {"course_id": course_id, "title": title}


def delete_course(course_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM courses WHERE course_id = ?", (course_id,))
        return cur.rowcount > 0


def list_students() -> list[dict]:
    with connect() as conn:
        students = conn.execute(
            "SELECT student_id, name FROM students ORDER BY student_id"
        ).fetchall()
        result = []
        for s in students:
            courses = [
                r["course_id"]
                for r in conn.execute(
                    """
                    SELECT course_id FROM enrollments
                    WHERE student_id = ? ORDER BY course_id
                    """,
                    (s["student_id"],),
                ).fetchall()
            ]
            result.append(
                {
                    "student_id": s["student_id"],
                    "name": s["name"],
                    "courses": courses,
                }
            )
    return result


def get_student(student_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT student_id, name FROM students WHERE student_id = ?",
            (student_id,),
        ).fetchone()
        if not row:
            return None
        courses = [
            r["course_id"]
            for r in conn.execute(
                """
                SELECT course_id FROM enrollments
                WHERE student_id = ? ORDER BY course_id
                """,
                (student_id,),
            ).fetchall()
        ]
    return {
        "student_id": row["student_id"],
        "name": row["name"],
        "courses": courses,
    }


def get_student_course_ids(student_id: str) -> list[str]:
    student = get_student(student_id)
    return student["courses"] if student else []


def upsert_student(student_id: str, name: str = "", courses: list[str] | None = None) -> dict:
    student_id = student_id.strip()
    name = (name or student_id).strip()
    courses = list(dict.fromkeys(c.strip() for c in (courses or []) if c.strip()))
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO students(student_id, name) VALUES (?, ?)
            ON CONFLICT(student_id) DO UPDATE SET name = excluded.name
            """,
            (student_id, name),
        )
        conn.execute("DELETE FROM enrollments WHERE student_id = ?", (student_id,))
        for course_id in courses:
            conn.execute(
                "INSERT OR IGNORE INTO courses(course_id, title) VALUES (?, ?)",
                (course_id, course_id),
            )
            conn.execute(
                "INSERT INTO enrollments(student_id, course_id) VALUES (?, ?)",
                (student_id, course_id),
            )
    return get_student(student_id)


def enroll(student_id: str, course_id: str) -> dict | None:
    student_id, course_id = student_id.strip(), course_id.strip()
    with connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO students(student_id, name) VALUES (?, ?)",
                (student_id, student_id),
            )
        conn.execute(
            "INSERT OR IGNORE INTO courses(course_id, title) VALUES (?, ?)",
            (course_id, course_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO enrollments(student_id, course_id) VALUES (?, ?)",
            (student_id, course_id),
        )
    return get_student(student_id)


def unenroll(student_id: str, course_id: str) -> dict | None:
    with connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if not exists:
            return None
        conn.execute(
            "DELETE FROM enrollments WHERE student_id = ? AND course_id = ?",
            (student_id, course_id),
        )
    return get_student(student_id)


def delete_student(student_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        return cur.rowcount > 0
