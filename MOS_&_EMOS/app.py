"""
TTS Naturalness (MOS) & Emotion (EMOS) Evaluation Study
---------------------------------------------------------
A Streamlit app for collecting 5-point Likert ratings on:
  - MOS  : naturalness of synthesized speech
  - EMOS : how well the speech conveys a target emotion

Design (per study spec):
  - 2 TTS models: Parler TTS, CosyVoice (never shown to the rater)
  - 4 emotions:   angry, happy, sad, surprise
  - 5 samples per (model x emotion) cell  -> 40 samples total
  - Each sample gets ONE MOS question + ONE EMOS question
  - Both questions must be answered before the user can move on
  - Progress is saved to a Google Sheet after every sample, keyed by the
    participant's name, so they can resume later (even in a new browser
    session, or after the app itself restarts) just by typing the same
    name again. Storing responses in Google Sheets (instead of a local
    CSV) means data survives redeploys/sleeps if you host this on
    Streamlit Community Cloud, whose local filesystem is ephemeral.

Run with:
    streamlit run app.py

See README.md for how to set up the Google Sheet + service account
credentials this app needs (via st.secrets).
"""

import hashlib
import random
from datetime import datetime
from pathlib import Path

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
import typing

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

AUDIO_ROOT = Path("audio_samples")

# Google Sheets config — see README.md for setup steps.
# `sheet_key` (the spreadsheet ID from its URL) and the service-account
# credentials live in st.secrets, never hard-coded here.
WORKSHEET_NAME = "responses"
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

# folder-name (on disk) -> display name
MODELS = {
    "parler_tts": "Parler TTS",
    "cosyvoice": "CosyVoice",
}
EMOTIONS = ["angry", "happy", "sad", "surprise"]
SAMPLES_PER_CELL = 5
TOTAL_SAMPLES = len(MODELS) * len(EMOTIONS) * SAMPLES_PER_CELL  # 40

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

RESPONSE_COLUMNS = [
    "user_name", "sample_id", "model", "emotion",
    "mos_score", "emos_score", "timestamp",
]

MOS_SCALE = {
    1: "1 — Completely unnatural (clearly robotic / synthetic)",
    2: "2 — Mostly unnatural",
    3: "3 — Neutral (neither natural nor unnatural)",
    4: "4 — Mostly natural",
    5: "5 — Completely natural (indistinguishable from a human speaker)",
}

EMOS_SCALE = {
    1: "1 — Emotion is not perceivable at all / wrong emotion is conveyed",
    2: "2 — Emotion is barely perceivable",
    3: "3 — Emotion is somewhat perceivable, but unclear or weak",
    4: "4 — Emotion is clearly perceivable and mostly appropriate",
    5: "5 — Emotion is fully, clearly, and appropriately conveyed",
}

st.set_page_config(page_title="Emotion Evaluation Study", page_icon="🎙️", layout="centered")


# --------------------------------------------------------------------------
# DATA / MANIFEST HELPERS
# --------------------------------------------------------------------------

def build_manifest():
    """Scan audio_samples/<model>/<emotion>/ for files and build the list
    of 40 samples. Returns (samples, problems)."""
    samples, problems = [], []
    for model_key in MODELS:
        for emotion in EMOTIONS:
            folder = AUDIO_ROOT / model_key / emotion
            if not folder.exists():
                problems.append(f"Missing folder: {folder}")
                continue
            files = sorted(
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
            )
            if len(files) < SAMPLES_PER_CELL:
                problems.append(
                    f"{folder} has {len(files)} audio file(s), needs {SAMPLES_PER_CELL}"
                )
            for f in files[:SAMPLES_PER_CELL]:
                samples.append({
                    "sample_id": f"{model_key}__{emotion}__{f.stem}",
                    "model": model_key,
                    "emotion": emotion,
                    "file_path": str(f),
                })
    return samples, problems


def get_user_order(username: str, samples: list):
    """Deterministic per-user shuffle, so resuming always reproduces the
    same presentation order for that participant."""
    seed = int(hashlib.sha256(username.strip().lower().encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    order = samples.copy()
    rng.shuffle(order)
    return order


def _open_spreadsheet(client: gspread.Client):
    """Open the target spreadsheet, by key if provided, else by name."""
    sheet_key = st.secrets.get("sheet_key")
    if sheet_key:
        # Accept either the raw spreadsheet ID or a full Google Sheets URL
        # (users sometimes paste the full URL into secrets). Normalize to
        # the spreadsheet ID so gspread.open_by_key works as expected.
        def _normalize_key(key: str) -> str:
            key = key.strip()
            if "/d/" in key:
                try:
                    return key.split("/d/")[1].split("/")[0]
                except Exception:
                    return key
            return key

        norm_key = _normalize_key(sheet_key)
        try:
            return client.open_by_key(norm_key)
        except Exception as e:  # surface a clearer message for common mistakes
            raise Exception(
                f"Could not open spreadsheet with key '{norm_key}'. "
                f"(original value: '{sheet_key}'). Check the ID and sharing: {e}"
            )
    sheet_name = st.secrets.get("sheet_name", "TTS Evaluation Responses")
    return client.open(sheet_name)


@st.cache_resource(show_spinner=False)
def get_worksheet():
    """Authorize against Google Sheets and return the worksheet that holds
    responses, creating it (and its header row) if it doesn't exist yet.
    Cached as a resource so we only pay the auth + setup cost once per
    running app process."""
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_info, scopes=SHEETS_SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = _open_spreadsheet(client)

    try:
        ws = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(
            title=WORKSHEET_NAME, rows=2000, cols=len(RESPONSE_COLUMNS)
        )

    existing = ws.get_all_values()
    if not existing:
        ws.append_row(RESPONSE_COLUMNS)
    elif existing[0] != RESPONSE_COLUMNS:
        ws.insert_row(RESPONSE_COLUMNS, index=1)

    return ws


def load_responses() -> pd.DataFrame:
    ws = get_worksheet()
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    df = pd.DataFrame.from_records(records)
    for col in RESPONSE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[RESPONSE_COLUMNS]


def save_response(row: dict):
    """Append a single rated sample as a new row in the Google Sheet."""
    ws = get_worksheet()
    values = [row.get(col, "") for col in RESPONSE_COLUMNS]
    ws.append_row(values, value_input_option="USER_ENTERED")


def get_completed_sample_ids(username: str) -> set:
    df = load_responses()
    if df.empty:
        return set()
    user_df = df[df["user_name"].astype(str).str.strip().str.lower() == username.strip().lower()]
    return set(user_df["sample_id"].tolist())


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------

def init_state():
    defaults = {
        "page": "landing",
        "user_name": "",
        "order": [],
        "current_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------

def landing_page():
    st.title("🎙️ Emotion Evaluation Study")
    st.write(
        "Thank you for taking part in this listening study! You will listen to "
        "short speech clips generated by two different text-to-speech (TTS) "
        "systems and answer two short questions about each one."
    )

    with st.expander("📋 Instructions", expanded=True):
        st.markdown(
            f"""
- You will evaluate **{TOTAL_SAMPLES} short audio clips** in total.
- Each clip was generated to express one of four emotions: **angry, happy, sad, surprise**.
- For every clip you will answer **two questions**:
  1. **MOS** — how *natural* (human-like) the speech sounds.
  2. **EMOS** — how well the speech conveys the *intended emotion*.
- Please listen to each clip **fully, and only once or twice**, before answering.
- **Both questions must be answered** before you can move to the next clip.
- The study takes roughly 15–20 minutes.
            """
        )

    with st.expander("📏 Rating Scales", expanded=False):
        st.markdown("**MOS — Naturalness**")
        for v in range(1, 6):
            st.write(MOS_SCALE[v])
        st.markdown("**EMOS — Emotion**")
        for v in range(1, 6):
            st.write(EMOS_SCALE[v])

    with st.expander("⚠️ Important Guidelines", expanded=False):
        st.markdown(
            """
- Please use **headphones** in a quiet environment if possible.
- Rate **naturalness and emotion independently** — a clip can sound natural but
  convey the wrong emotion, or vice versa.
- Base EMOS purely on what you *hear*, not on what emotion you'd expect from the
  clip's label.
- Try to rate consistently across the session; avoid rushing.
- Do **not** close the browser tab while a rating is in progress — once you
  click "Next", your answer is saved automatically and you can safely leave
  and resume later.
- To resume, simply re-enter the **exact same name** you used originally.
            """
        )

    st.divider()
    st.subheader("Get started")

    sheets_ready, sheets_error = True, None
    try:
        get_worksheet()
    except Exception as e:  # noqa: BLE001 - surface a friendly setup message
        sheets_ready, sheets_error = False, str(e)

    if not sheets_ready:
        st.error(
            "Couldn't connect to the Google Sheet used to store responses. "
            "Check that `gcp_service_account` and `sheet_key` (or `sheet_name`) "
            "are set in `.streamlit/secrets.toml` (or your Streamlit Cloud app's "
            "Secrets settings), and that the sheet is shared with the service "
            f"account's email.\n\nDetails: {sheets_error}"
        )

    name = st.text_input("Enter your name", value=st.session_state.user_name,
                          placeholder="e.g., Simi Sharma")

    samples, problems = build_manifest()
    if problems:
        st.error(
            "The audio library isn't fully set up yet. Please add the missing "
            "files before starting:\n\n" + "\n".join(f"- {p}" for p in problems)
        )

    start_disabled = bool(problems) or not sheets_ready or not name.strip()
    if st.button("Start / Resume ➡️", type="primary", disabled=start_disabled):
        clean_name = name.strip()
        order = get_user_order(clean_name, samples)
        completed = get_completed_sample_ids(clean_name)
        # order is deterministic per user, so completed samples are always
        # a prefix of `order` as long as the participant always proceeds
        # forward (which the UI enforces).
        completed_count = sum(1 for s in order if s["sample_id"] in completed)

        st.session_state.user_name = clean_name
        st.session_state.order = order
        st.session_state.current_index = completed_count
        st.session_state.page = "survey" if completed_count < len(order) else "done"
        st.rerun()


def survey_page():
    order = st.session_state.order
    idx = st.session_state.current_index
    total = len(order)
    sample = order[idx]

    st.progress(idx / total, text=f"Sample {idx + 1} of {total}")
    st.caption(f"Logged in as **{st.session_state.user_name}**")

    st.subheader(f"Sample {idx + 1} of {total}")
    st.write(f"**Target emotion:** {sample['emotion'].capitalize()}")

    st.audio(sample["file_path"])

    st.markdown("**Q1. Naturalness (MOS)** — How natural does this audio sound?")
    mos = st.radio(
        "MOS rating", options=[1, 2, 3, 4, 5],
        format_func=lambda v: MOS_SCALE[v],
        index=None, key=f"mos_{sample['sample_id']}",
        label_visibility="collapsed",
    )

    st.markdown(
        f"**Q2. Emotion (EMOS)** — How well does this audio convey **{sample['emotion']}**?"
    )
    emos = st.radio(
        "EMOS rating", options=[1, 2, 3, 4, 5],
        format_func=lambda v: EMOS_SCALE[v],
        index=None, key=f"emos_{sample['sample_id']}",
        label_visibility="collapsed",
    )

    st.write("")
    if st.button("Next ➡️", type="primary"):
        if mos is None or emos is None:
            st.warning("Please answer both questions before continuing.")
        else:
            save_response({
                "user_name": st.session_state.user_name,
                "sample_id": sample["sample_id"],
                "model": sample["model"],
                "emotion": sample["emotion"],
                "mos_score": mos,
                "emos_score": emos,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            st.session_state.current_index += 1
            if st.session_state.current_index >= total:
                st.session_state.page = "done"
            st.rerun()


def done_page():
    st.balloons()
    st.title("✅ All done — thank you!")
    st.write(
        f"Thank you, **{st.session_state.user_name}**! You've completed all "
        f"{TOTAL_SAMPLES} ratings for this study."
    )
    st.write("You may now close this window.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    init_state()
    page = st.session_state.page
    if page == "landing":
        landing_page()
    elif page == "survey":
        survey_page()
    else:
        done_page()


if __name__ == "__main__":
    main()
