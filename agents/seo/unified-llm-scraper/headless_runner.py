import os
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ai_models import chatgpt, perplexity, gemini, grok, copilot, AIModelRetriever
from db import (
    save_run,
    list_prompts,
    claim_due_schedule,
    advance_schedule_after_run,
)

OUTPUT_DIR = Path("output")
MAX_WORKERS = 5


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def write_output(model_name: str, payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{sanitize_filename(model_name)}-output.json"
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    return path


def extract_answer_text(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("answer_text"), str):
        return payload["answer_text"]

    # fallback: some datasets return HTML
    if isinstance(payload.get("answer_html"), str):
        return payload["answer_html"]

    if "data" in payload and isinstance(payload["data"], list) and payload["data"]:
        first = payload["data"][0]
        if isinstance(first, dict) and isinstance(first.get("answer_text"), str):
            return first["answer_text"]

    return None


def mentions_target(payload: dict, target_phrase: str) -> bool:
    if not target_phrase:
        return False

    answer = extract_answer_text(payload)
    if isinstance(answer, str):
        return target_phrase.lower() in answer.lower()

    # fallback: scan full payload
    try:
        blob = json.dumps(payload, ensure_ascii=False)
        return target_phrase.lower() in blob.lower()
    except Exception:
        return False

def persist_run(*, model_name: str, prompt: str, payload, target_phrase: str, country: str = "") -> bool:
    if payload is None:
        print(f"{model_name}: skipping DB insert (payload is None).")
        return False

    # If you want to treat empty list/dict as "don't save", keep this:
    if payload == {} or payload == []:
        print(f"{model_name}: skipping DB insert (empty payload). type={type(payload).__name__}")
        return False

    try:
        json.dumps(payload, ensure_ascii=False)
    except TypeError as e:
        print(f"{model_name}: payload not JSON-serializable ({e}). Stringifying.")
        payload = {"raw": json.dumps(payload, default=str, ensure_ascii=False)}

    mentioned = mentions_target(payload if isinstance(payload, dict) else {"data": payload}, target_phrase)

    try:
        save_run(
            model_name=model_name,
            prompt=prompt,
            country=country,
            target_phrase=target_phrase,
            mentioned=mentioned,
            payload=payload,
        )
    except Exception as db_err:
        print(f"{model_name}: DB insert failed: {db_err}")

    return mentioned



def run_schedule_once(
    *,
    schedule_row: dict,
    retriever: AIModelRetriever,
    available_models: list,
    model_by_name: dict,
    save_to_disk: bool = False,
) -> None:
    """
    Executes a single schedule row:
      - loads prompts (active-only if configured)
      - runs prompts × models
      - saves results to DB
      - advances schedule
    """
    sched_id = schedule_row["id"]
    sched_name = schedule_row.get("name") or f"schedule-{sched_id}"
    sched_models = schedule_row.get("models") or []
    sched_country = schedule_row.get("country") or ""
    sched_target = schedule_row.get("target_phrase") or ""
    only_active = bool(schedule_row.get("only_active_prompts"))

    prompt_rows = list_prompts(only_active=only_active)
    prompts_list = [p["prompt"] for p in prompt_rows]

    if not prompts_list:
        print(f"[{sched_name}] No prompts to run (only_active={only_active}). Advancing schedule.")
        advance_schedule_after_run(schedule_id=sched_id)
        return

    # filter models to ones we actually know about
    sched_models = [m for m in sched_models if m in model_by_name]
    if not sched_models:
        print(f"[{sched_name}] No valid models configured on schedule. Advancing schedule.")
        advance_schedule_after_run(schedule_id=sched_id)
        return

    print(f"[{sched_name}] Running {len(prompts_list)} prompt(s) × {len(sched_models)} model(s)")

    for pr in prompts_list:
        print(f"[{sched_name}] Prompt: {pr}")

        def run_one(model_name: str):
            model = model_by_name[model_name]
            payload = retriever.run(model, pr, country=sched_country)
            return model_name, payload

        # run all models concurrently for this prompt
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(sched_models))) as pool:
            futures = [pool.submit(run_one, m) for m in sched_models]

            for fut in as_completed(futures):
                try:
                    model_name, payload = fut.result()
                except Exception as e:
                    print(f"[{sched_name}] Model task failed: {e}")
                    continue
                
                persist_run(
                    model_name=model_name,
                    prompt=pr,
                    payload=payload,
                    target_phrase=sched_target,
                    country=sched_country,
                )

                if save_to_disk and isinstance(payload, dict) and payload:
                    try:
                        write_output(model_name, payload)
                    except Exception as e:
                        print(f"[{sched_name}] Failed writing output for {model_name}: {e}")

    # IMPORTANT: only advance after the work completes
    advance_schedule_after_run(schedule_id=sched_id)
    print(f"[{sched_name}] Completed and advanced.")


def main():
    load_dotenv()

    api_token = os.getenv("BRIGHTDATA_API_TOKEN")
    if not api_token:
        raise RuntimeError("Missing BRIGHTDATA_API_TOKEN in environment/.env")

    retriever = AIModelRetriever(api_token=api_token)

    available_models = [chatgpt, perplexity, gemini, grok, copilot]
    model_by_name = {m.name: m for m in available_models}

    # runner identity for locking
    lock_owner = f"runner-{uuid.uuid4()}"
    print(f"Headless runner started. lock_owner={lock_owner}")

    # tune these without DB changes
    tick_every_seconds = int(os.getenv("SCHED_TICK_SECONDS", "15"))      # how often to wake up
    lock_seconds = int(os.getenv("SCHED_LOCK_SECONDS", "1800"))         # lock duration while a job runs
    drain_all_due = os.getenv("SCHED_DRAIN_ALL_DUE", "1") == "1"         # run all due jobs each tick
    save_to_disk = os.getenv("SCHED_SAVE_TO_DISK", "0") == "1"

    while True:
        now_ts = int(time.time())

        ran_any = False

        # claim & run either one schedule, or drain all due schedules
        while True:
            try:
                due = claim_due_schedule(now_ts=now_ts, lock_owner=lock_owner, lock_seconds=lock_seconds)
            except Exception as e:
                print(f"Failed to claim due schedule: {e}")
                due = None

            if not due:
                break

            ran_any = True
            try:
                run_schedule_once(
                    schedule_row=due,
                    retriever=retriever,
                    available_models=available_models,
                    model_by_name=model_by_name,
                    save_to_disk=save_to_disk,
                )
            except Exception as e:
                # If something explodes mid-run, we do NOT advance the schedule.
                # The lock will expire, and the schedule will be picked up later.
                print(f"Schedule run crashed: {e}")

            if not drain_all_due:
                break

            # update time for next claim
            now_ts = int(time.time())

        if not ran_any:
            # optional: quieter logs
            print(f"[{int(time.time())}] No due schedules.")

        time.sleep(tick_every_seconds)


if __name__ == "__main__":
    main()
