# db.py
import os
from typing import Any, Optional
import time

from supabase import create_client, Client


def get_db() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_TOKEN")  # keep consistent with your .env
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_API_TOKEN in environment.")
    return create_client(url, key)


def save_run(
    *,
    model_name: str,
    prompt: str,
    country: str,
    target_phrase: str,
    mentioned: bool,
    payload: dict,
) -> dict:
    
    
    db = get_db()

    row = {
        "created_at_ts": int(time.time()),
        "model_name": model_name,
        "prompt": prompt,
        "country": country or None,
        "target_phrase": target_phrase or None,
        "mentioned": bool(mentioned),
        "payload": payload,  # JSONB
    }

    res = db.table("llm_runs").insert(row).execute()
    if not getattr(res, "data", None):
        row["payload"] = {"ERROR": "FAILED RUN"}
        res = db.table("llm_runs").insert(row).execute()

        raise RuntimeError(f"Insert failed: {res}")
    return res.data[0]


# -----------------------
# Prompt tracking
# -----------------------

def add_prompt(*, prompt: str, is_active: bool = True) -> dict:
    db = get_db()
    row = {
        "created_at_ts": int(time.time()),
        "prompt": prompt.strip(),
        "is_active": bool(is_active)
        }
    res = db.table("prompts").insert(row).execute()
    if not getattr(res, "data", None):
        raise RuntimeError(f"Insert failed: {res}")
    return res.data[0]


def list_prompts(*, only_active: bool = False, limit: int = 500) -> list[dict]:
    db = get_db()
    q = db.table("prompts").select("*").order("created_at_ts", desc=True).limit(limit)
    if only_active:
        q = q.eq("is_active", True)
    res = q.execute()
    return res.data or []


def set_prompt_active(*, prompt_id: int, is_active: bool) -> None:
    db = get_db()
    res = db.table("prompts").update({"is_active": bool(is_active)}).eq("id", prompt_id).execute()
    if res.data is None:
        raise RuntimeError(f"Update failed: {res}")


# -----------------------
# Search / filter history
# -----------------------

def list_runs_filtered(
    *,
    limit: int = 200,
    model_name: str | None = None,
    mentioned: bool | None = None,
    target_phrase: str | None = None,
    prompt_contains: str | None = None,
) -> list[dict]:
    db = get_db()
    q = db.table("llm_runs").select("*").order("created_at_ts", desc=True).limit(limit)

    if model_name:
        q = q.eq("model_name", model_name)

    if mentioned is not None:
        q = q.eq("mentioned", bool(mentioned))

    if target_phrase:
        q = q.eq("target_phrase", target_phrase)

    if prompt_contains:
        # Supabase PostgREST "like"
        q = q.ilike("prompt", f"%{prompt_contains}%")

    res = q.execute()
    return res.data or []


def distinct_values(column: str, *, limit: int = 200) -> list[str]:
    db = get_db()
    res = db.table("llm_runs").select(column).order("created_at_ts", desc=True).limit(limit).execute()
    vals = []
    for row in (res.data or []):
        v = row.get(column)
        if v and v not in vals:
            vals.append(v)
    return vals


# -----------------------
# Reporting (basic)
# -----------------------

def mention_rate_by_model(*, target_phrase: str, limit: int = 5000) -> list[dict[str, Any]]:
    """
    Returns rows like: {"model_name": "ChatGPT", "total": 123, "mentions": 45, "rate": 0.3658}
    Computed client-side for simplicity.
    """
    rows = list_runs_filtered(limit=min(limit, 5000), target_phrase=target_phrase)
    totals: dict[str, int] = {}
    mentions: dict[str, int] = {}

    for r in rows:
        m = r.get("model_name") or "Unknown"
        totals[m] = totals.get(m, 0) + 1
        if r.get("mentioned") is True:
            mentions[m] = mentions.get(m, 0) + 1

    out = []
    for m in sorted(totals.keys()):
        t = totals[m]
        yes = mentions.get(m, 0)
        out.append({"model_name": m, "total": t, "mentions": yes, "rate": (yes / t) if t else 0.0})
    return out



def create_schedule(*, name: str, next_run_ts: int, models: list[str],
                    country: str = "", target_phrase: str = "", only_active_prompts: bool = True,
                    is_enabled: bool = True, repeat_every_seconds: int = 86400) -> dict:
    """
    repeat_every_seconds: default 86400 for daily. Set to 0 for 'one-shot' schedules.
    """
    db = get_db()
    row = {
        "name": name.strip(),
        "is_enabled": bool(is_enabled),
        "next_run_ts": int(next_run_ts),
        "last_run_ts": None,
        "models": models,  # jsonb array
        "country": country or None,
        "target_phrase": target_phrase or None,
        "only_active_prompts": bool(only_active_prompts),
        "locked_until_ts": None,
        "lock_owner": None,
        "repeat_every_seconds": int(repeat_every_seconds),
    }
    res = db.table("schedules").insert(row).execute()
    if not getattr(res, "data", None):
        raise RuntimeError(f"Insert failed: {res}")
    return res.data[0]


def list_schedules(*, limit: int = 200) -> list[dict]:
    db = get_db()
    res = db.table("schedules").select("*").order("next_run_ts", desc=False).limit(limit).execute()
    return res.data or []


def set_schedule_enabled(*, schedule_id: int, is_enabled: bool) -> None:
    db = get_db()
    res = db.table("schedules").update({"is_enabled": bool(is_enabled)}).eq("id", schedule_id).execute()
    if res.data is None:
        raise RuntimeError(f"Update failed: {res}")


def delete_schedule(*, schedule_id: int) -> None:
    db = get_db()
    res = db.table("schedules").delete().eq("id", schedule_id).execute()
    if res.data is None:
        raise RuntimeError(f"Delete failed: {res}")


def claim_due_schedule(*, now_ts: Optional[int] = None, lock_owner: str,
                       lock_seconds: int = 900) -> Optional[dict]:
    """
    Best-effort claim:
      - find earliest due schedule
      - try to lock it (guarded update)
      - return claimed row or None

    NOTE: This is not perfectly atomic without an RPC, but it’s good enough for “single app open”
    and the lock prevents duplicates across tabs in most cases.
    """
    db = get_db()
    now_ts = int(now_ts or time.time())

    # find candidates due now
    res = (
        db.table("schedules")
        .select("*")
        .eq("is_enabled", True)
        .lte("next_run_ts", now_ts)
        .order("next_run_ts", desc=False)
        .limit(10)
        .execute()
    )
    candidates = res.data or []
    if not candidates:
        return None

    for s in candidates:
        sid = s["id"]
        locked_until = s.get("locked_until_ts")

        if locked_until is not None and int(locked_until) > now_ts:
            continue

        lock_until_ts = now_ts + int(lock_seconds)

        # guarded update (only lock if still due and lock is free/expired)
        q = (
            db.table("schedules")
            .update({"locked_until_ts": lock_until_ts, "lock_owner": lock_owner})
            .eq("id", sid)
            .eq("is_enabled", True)
            .lte("next_run_ts", now_ts)
        )
        # if it was unlocked, locked_until_ts may be null; if expired, it will be <= now_ts
        # PostgREST limitations make an OR tricky; we just re-check after update.
        upd = q.execute()
        if getattr(upd, "data", None):
            # refetch and verify lock_owner matches
            got = db.table("schedules").select("*").eq("id", sid).limit(1).execute().data
            if got and got[0].get("lock_owner") == lock_owner:
                return got[0]

    return None


def advance_schedule_after_run(*, schedule_id: int, now_ts: Optional[int] = None) -> dict:
    """
    - set last_run_ts
    - advance next_run_ts if recurring, else disable
    - clear lock
    """
    db = get_db()
    now_ts = int(now_ts or time.time())

    # fetch schedule
    s = db.table("schedules").select("*").eq("id", schedule_id).limit(1).execute().data
    if not s:
        raise RuntimeError("Schedule not found")
    s = s[0]

    repeat = int(s.get("repeat_every_seconds") or 0)
    if repeat > 0:
        next_run = int(s["next_run_ts"]) + repeat
        updates = {"last_run_ts": now_ts, "next_run_ts": next_run, "locked_until_ts": None, "lock_owner": None}
    else:
        updates = {"last_run_ts": now_ts, "is_enabled": False, "locked_until_ts": None, "lock_owner": None}

    res = db.table("schedules").update(updates).eq("id", schedule_id).execute()
    if not getattr(res, "data", None):
        raise RuntimeError(f"Update failed: {res}")
    return res.data[0]

def count_runs_since(
    *,
    since_ts: int,
    target_phrase: str,
    country: str,
    limit: int = 5000,
) -> list[dict]:
    db = get_db()
    q = (
        db.table("llm_runs")
        .select("created_at_ts,model_name,prompt,country,target_phrase")
        .gte("created_at_ts", since_ts)
        .eq("target_phrase", target_phrase or None)
        .order("created_at_ts", desc=True)
        .limit(min(limit, 5000))
    )
    if country:
        q = q.eq("country", country)
    else:
        q = q.is_("country", "null")  # stored as None
    res = q.execute()
    return res.data or []
