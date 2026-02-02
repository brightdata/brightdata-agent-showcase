# app.py
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import time
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ai_models import chatgpt, perplexity, gemini, grok, copilot, AIModelRetriever
from db import (
    save_run,
    add_prompt,
    list_prompts,
    set_prompt_active,
    list_runs_filtered,
    distinct_values,
    mention_rate_by_model,
    # scheduler
    get_db,
    create_schedule,
    list_schedules,
    set_schedule_enabled,
    delete_schedule,
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


def main():
    st.set_page_config(page_title="Universal LLM Scraper", layout="wide")
    st.title("Universal LLM Scraper")

    load_dotenv()

    api_token = os.getenv("BRIGHTDATA_API_TOKEN")
    if not api_token:
        st.error("Missing BRIGHTDATA_API_TOKEN. Add it to a .env file in the project root.")
        st.stop()

    # Build models
    models = [chatgpt, perplexity, gemini, grok, copilot]
    model_names = [m.name for m in models]
    model_by_name = {m.name: m for m in models}

    # Session state
    st.session_state.setdefault("results", {})  # model_name -> payload
    st.session_state.setdefault("errors", {})   # key -> error
    st.session_state.setdefault("paths", {})    # model_name -> path

    # Tabs
    run_tab, prompts_tab, history_tab, reports_tab, scheduler_tab = st.tabs(
        ["Run", "Prompts", "History", "Reports", "Scheduler"]
    )

    # ----------------------------
    # Run tab
    # ----------------------------
    with run_tab:
        with st.sidebar:
            st.header("Run settings")
            prompt = st.text_area("Prompt", value="Who are the best residential proxy providers?", height=120)
            target_phrase = st.text_input("Target phrase to track", value="Bright Data")
            selected = st.multiselect("Models", options=model_names, default=model_names)
            country = st.text_input("Country (optional)", value="")
            save_to_disk = st.checkbox("Save results to output/", value=True)

            redact_terms = st.text_area("Brand terms to hide (one per line)", value="")
            redact_mode = st.selectbox("Hide mode", ["Mask", "Remove"], index=0)

            run_clicked = st.button("Run scrapes", type="primary", width="content")

        def apply_redaction(text: str) -> str:
            terms = [t.strip() for t in redact_terms.splitlines() if t.strip()]
            if not terms:
                return text
            pattern = re.compile(r"(" + "|".join(map(re.escape, terms)) + r")", flags=re.IGNORECASE)
            if redact_mode == "Mask":
                return pattern.sub("███", text)
            return pattern.sub("", text)

        def mentions_target(payload: dict) -> bool:
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

        def persist_run(model_name: str, payload: dict) -> bool:
            mentioned = mentions_target(payload)
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
                st.warning(f"{model_name}: DB insert failed: {db_err}")
            return mentioned

        status_col, results_col = st.columns([1, 2], gap="large")

        with status_col:
            st.subheader("Status")

            if run_clicked:
                st.session_state.results = {}
                st.session_state.errors = {}
                st.session_state.paths = {}

                if not selected:
                    st.warning("Select at least one model.")
                    st.stop()

                retriever = AIModelRetriever(api_token=api_token)

                status_boxes = {name: st.empty() for name in selected}
                progress = st.progress(0.0)
                done = 0
                total = len(selected)

                def run_one(model_name: str):
                    model = model_by_name[model_name]
                    payload = retriever.run(model, prompt, country=country)
                    return model_name, payload

                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as pool:
                    futures = [pool.submit(run_one, name) for name in selected]

                    for fut in as_completed(futures):
                        try:
                            model_name, payload = fut.result()
                            st.session_state.results[model_name] = payload

                            persist_run(model_name, payload)

                            status_boxes[model_name].success(f"{model_name}: done")
                            if save_to_disk:
                                path = write_output(model_name, payload)
                                st.session_state.paths[model_name] = str(path)

                        except Exception as e:
                            err = str(e)
                            st.session_state.errors[f"job-{done+1}"] = err
                            st.error(err)

                        done += 1
                        progress.progress(done / total)

                st.success("Run complete.")

            if st.session_state.paths:
                st.caption("Saved files")
                for k, v in st.session_state.paths.items():
                    st.write(f"- {k}: {v}")

            if st.session_state.errors:
                st.caption("Errors")
                for k, v in st.session_state.errors.items():
                    st.write(f"- {k}: {v}")

        with results_col:
            st.subheader("Results")

            if not st.session_state.results:
                st.info("Use the sidebar and click “Run scrapes” to collect results.")
            else:
                tabs = st.tabs(list(st.session_state.results.keys()))
                for tab, model_name in zip(tabs, st.session_state.results.keys()):
                    payload = st.session_state.results[model_name]
                    with tab:
                        mentioned = mentions_target(payload)
                        st.markdown(f"**Target phrase mentioned:** {'✅' if mentioned else '❌'}")

                        answer_text = extract_answer_text(payload)
                        if answer_text:
                            st.markdown("### Answer")
                            st.text_area(label="", value=apply_redaction(answer_text), height=260)
                        else:
                            st.markdown("### Raw JSON")
                            st.json(payload)

    # ----------------------------
    # Prompts tab
    # ----------------------------
    with prompts_tab:
        st.subheader("Prompt library")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Add prompts")
            bulk = st.text_area(
                "One prompt per line",
                value="Who are the best residential proxy providers?\nWhat is the best SERP API in 2026?",
                height=140,
            )
            if st.button("Save prompts", type="primary"):
                lines = [l.strip() for l in bulk.splitlines() if l.strip()]
                saved = 0
                for line in lines:
                    try:
                        add_prompt(prompt=line, is_active=True)
                        saved += 1
                    except Exception as e:
                        st.warning(f"Failed to save prompt: {line[:60]}... ({e})")
                st.success(f"Saved {saved} prompts.")

        with right:
            st.markdown("### Existing prompts")
            only_active = st.checkbox("Show active only", value=False)
            prompts = list_prompts(only_active=only_active)

            if not prompts:
                st.info("No prompts saved yet.")
            else:
                for p in prompts:
                    c1, c2 = st.columns([10, 2])
                    with c1:
                        st.write(p["prompt"])
                    with c2:
                        # Icon-only toggle (no label text to wrap)
                        new_active = st.toggle(
                            "Active",
                            value=bool(p["is_active"]),
                            key=f"prompt-active-{p['id']}",
                            help="Enable/disable this prompt for bulk runs",
                            label_visibility="collapsed",
                        )


                        # Persist only when changed
                        if new_active != bool(p["is_active"]):
                            set_prompt_active(prompt_id=p["id"], is_active=new_active)
                            st.rerun()


        st.divider()
        st.markdown("### Bulk run (all active prompts)")

        bulk_target = st.text_input("Target phrase for bulk run", value="Bright Data", key="bulk-target")
        bulk_country = st.text_input("Country (optional)", value="", key="bulk-country")
        bulk_models = st.multiselect("Models", options=model_names, default=model_names, key="bulk-models")

        if st.button("Run active prompts now", type="primary"):
            active_prompts = [p["prompt"] for p in list_prompts(only_active=True)]
            if not active_prompts:
                st.warning("No active prompts found. Enable at least one in the prompt library.")
                st.stop()

            retriever = AIModelRetriever(api_token=api_token)

            st.info(f"Running {len(active_prompts)} prompt(s) × {len(bulk_models)} model(s)")

            for pr in active_prompts:
                st.markdown(f"#### Prompt: {pr}")
                status = st.empty()

                def mentions(payload: dict) -> bool:
                    ans = extract_answer_text(payload) or ""
                    return bulk_target.lower() in ans.lower()

                def run_one(model_name: str):
                    model = model_by_name[model_name]
                    payload = retriever.run(model, pr, country=bulk_country)
                    return model_name, payload

                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(bulk_models))) as pool:
                    futures = [pool.submit(run_one, m) for m in bulk_models]

                    for fut in as_completed(futures):
                        model_name, payload = fut.result()
                        try:
                            save_run(
                                model_name=model_name,
                                prompt=pr,
                                country=bulk_country,
                                target_phrase=bulk_target,
                                mentioned=mentions(payload),
                                payload=payload,
                            )
                        except Exception as e:
                            st.warning(f"{model_name}: DB insert failed ({e})")

                status.success("Saved bulk run results to DB.")

    # ----------------------------
    # History tab
    # ----------------------------
    with history_tab:
        st.subheader("Run history")

        models_in_db = ["(any)"] + distinct_values("model_name")
        targets_in_db = ["(any)"] + distinct_values("target_phrase")

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            model_filter = st.selectbox("Model", models_in_db)
        with c2:
            target_filter = st.selectbox("Target phrase", targets_in_db)
        with c3:
            mention_filter = st.selectbox("Mentioned?", ["(any)", "yes", "no"])
        with c4:
            prompt_contains = st.text_input("Prompt contains", value="")

        limit = st.slider("Rows", min_value=50, max_value=500, value=200, step=50)

        rows = list_runs_filtered(
            limit=limit,
            model_name=None if model_filter == "(any)" else model_filter,
            target_phrase=None if target_filter == "(any)" else target_filter,
            mentioned=None if mention_filter == "(any)" else (mention_filter == "yes"),
            prompt_contains=prompt_contains or None,
        )

        if not rows:
            st.info("No matching runs.")
        else:
            df = pd.DataFrame(rows)
            df["created_at"] = df["created_at_ts"].apply(
                lambda x: datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S")
            )
            slim = df[["created_at", "model_name", "mentioned", "target_phrase", "prompt"]].copy()
            st.dataframe(slim, width="content", hide_index=True)

            st.markdown("### Inspect a row")
            idx = st.number_input(
                "Row index (0-based in the table above)",
                min_value=0,
                max_value=len(rows) - 1,
                value=0,
            )
            st.json(rows[int(idx)].get("payload"))

    # ----------------------------
    # Reports tab
    # ----------------------------
    with reports_tab:
        st.subheader("Reports")

        report_target = st.text_input("Target phrase for report", value="Bright Data", key="report-target")
        stats = mention_rate_by_model(target_phrase=report_target)

        if not stats:
            st.info("No data for that target phrase yet.")
        else:
            df = pd.DataFrame(stats)
            df["rate"] = (df["rate"] * 100.0).round(2)
            st.dataframe(df, width="content", hide_index=True)

            st.markdown("### Mention rate (percent)")
            chart = df.set_index("model_name")["rate"]
            st.bar_chart(chart)

    # ----------------------------
    # Scheduler tab
    # ----------------------------
    with scheduler_tab:
        
        st.markdown("### Create a schedule")

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sched_name = st.text_input("Name", value="Daily check", key="sched-name")
        with c2:
            sched_date = st.date_input("Date", value=datetime.now().date(), key="sched-date")
        with c3:
            sched_time = st.time_input(
                "Time (local)",
                value=datetime.now().time().replace(second=0, microsecond=0),
                key="sched-time",
            )

        c4, c5 = st.columns([2, 1])
        with c4:
            sched_models = st.multiselect(
                "Models",
                options=model_names,
                default=model_names,
                key="sched-models",
            )
        with c5:
            repeat_daily = st.checkbox("Repeat daily", value=True, key="sched-repeat-daily")

        c6, c7, c8 = st.columns([1, 1, 1])
        with c6:
            sched_target = st.text_input("Target phrase", value="Bright Data", key="sched-target")
        with c7:
            sched_country = st.text_input("Country (optional)", value="", key="sched-country")
        with c8:
            sched_only_active = st.checkbox("Use active prompts only", value=True, key="sched-only-active")

        repeat_every_seconds = 86400 if repeat_daily else 0

        def compute_next_run_ts(date_obj, time_obj) -> int:
            dt_local = datetime.combine(date_obj, time_obj)
            ts = int(dt_local.timestamp())
            now_ts = int(time.time())

            if ts <= now_ts:
                if repeat_every_seconds > 0:
                    while ts <= now_ts:
                        ts += repeat_every_seconds
                else:
                    ts = now_ts
            return ts

        if st.button("Create schedule", type="primary", key="sched-create"):
            if not sched_models:
                st.warning("Pick at least one model.")
                st.stop()

            try:
                next_run_ts = compute_next_run_ts(sched_date, sched_time)
                create_schedule(
                    name=sched_name.strip() or "Schedule",
                    next_run_ts=next_run_ts,
                    models=sched_models,
                    country=sched_country,
                    target_phrase=sched_target,
                    only_active_prompts=sched_only_active,
                    is_enabled=True,
                    repeat_every_seconds=repeat_every_seconds,
                )
                st.success("Schedule created.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create schedule: {e}")

        st.divider()
        st.markdown("### Existing schedules")

        try:
            schedules = list_schedules(limit=200)
        except Exception as e:
            st.error(f"Failed to load schedules: {e}")
            schedules = []

        if not schedules:
            st.info("No schedules yet.")
        else:
            for s in schedules:
                next_dt = datetime.fromtimestamp(int(s["next_run_ts"]))
                last_dt = datetime.fromtimestamp(int(s["last_run_ts"])) if s.get("last_run_ts") else None
                repeat = int(s.get("repeat_every_seconds") or 0)

                box = st.container()
                with box:
                    cols = st.columns([3, 1, 1, 1])
                    with cols[0]:
                        st.write(f"**{s['name']}**")
                        st.caption(f"Next: {next_dt.strftime('%Y-%m-%d %H:%M:%S')} (local)")
                        st.caption(
                            f"Repeat: {'daily' if repeat == 86400 else ('off' if repeat == 0 else str(repeat) + 's')}"
                        )
                        st.caption(
                            f"Models: {', '.join(s.get('models') or [])}  |  "
                            f"Target: {s.get('target_phrase') or '(none)'}  |  "
                            f"Country: {s.get('country') or '(none)'}  |  "
                            f"Active prompts only: {'yes' if s.get('only_active_prompts') else 'no'}"
                        )
                        if last_dt:
                            st.caption(f"Last run: {last_dt.strftime('%Y-%m-%d %H:%M:%S')} (local)")

                    with cols[1]:
                        enabled = st.checkbox("Enabled", value=bool(s["is_enabled"]), key=f"sched-enabled-{s['id']}")
                        if enabled != bool(s["is_enabled"]):
                            try:
                                set_schedule_enabled(schedule_id=s["id"], is_enabled=enabled)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to update schedule: {e}")

                    with cols[2]:
                        if st.button("Delete", key=f"sched-del-{s['id']}"):
                            try:
                                delete_schedule(schedule_id=s["id"])
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")

                    with cols[3]:
                        if st.button("Run now", key=f"sched-runnow-{s['id']}"):
                            try:
                                db = get_db()
                                db.table("schedules").update(
                                    {"next_run_ts": int(time.time()), "locked_until_ts": None, "lock_owner": None}
                                ).eq("id", s["id"]).execute()
                                st.success("Marked as due. It will run on the next tick.")
                            except Exception as e:
                                st.error(f"Failed to mark due: {e}")

                st.divider()

        

if __name__ == "__main__":
    main()
