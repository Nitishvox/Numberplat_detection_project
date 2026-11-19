import io
import logging
import os
import re
import time

import cv2
import easyocr
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image


def get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = get_secret("ROBOFLOW_WORKSPACE", "m-nitish-46wkd")
ROBOFLOW_WORKFLOW_ID = get_secret(
    "ROBOFLOW_WORKFLOW_ID",
    "license-plate-recognition-vlicense-plate-recognition-rxg4e-l8dkk-1-yolo26n-t1-logic",
)
ROBOFLOW_API_URL = get_secret("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plate_app")

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh",
    "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli", "GA": "Goa",
    "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh", "JH": "Jharkhand",
    "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
    "OR": "Odisha (older code)", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}

# Characters OCR commonly confuses. Each map is applied only where the plate format
# needs a digit (LETTER_TO_DIGIT) or a letter (DIGIT_TO_LETTER).
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T"}

# OCR often reads the "IND" emblem printed on the left strip of the plate.
EMBLEM_PREFIXES = ("IND", "INO", "IN0", "1ND", "IN", "")


class PlateDetectionError(Exception):
    pass


@st.cache_resource
def get_client():
    if not ROBOFLOW_API_KEY:
        st.error(
            "ROBOFLOW_API_KEY is not set. Add it in .streamlit/secrets.toml "
            "or as an environment variable."
        )
        st.stop()
    return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def run_workflow_with_retry(client, image_pil, retries=2, backoff=1.5):
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": image_pil},
                parameters={},
            )
            return result[0]
        except Exception as error:
            last_error = error
            wait = backoff ** attempt
            logger.warning("Workflow call failed (attempt %s/%s): %s", attempt + 1, retries + 1, error)
            time.sleep(wait)
    raise PlateDetectionError(f"Workflow call failed after {retries + 1} attempts: {last_error}")


def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    def iou(first, second):
        f_x1, f_y1 = first["x"] - first["width"] / 2, first["y"] - first["height"] / 2
        f_x2, f_y2 = first["x"] + first["width"] / 2, first["y"] + first["height"] / 2
        s_x1, s_y1 = second["x"] - second["width"] / 2, second["y"] - second["height"] / 2
        s_x2, s_y2 = second["x"] + second["width"] / 2, second["y"] + second["height"] / 2
        intersection = max(0, min(f_x2, s_x2) - max(f_x1, s_x1))
        intersection *= max(0, min(f_y2, s_y2) - max(f_y1, s_y1))
        union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
        return intersection / (union + 1e-6)

    kept = []
    for box in sorted(boxes, key=lambda b: -b["confidence"]):
        if all(iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


# ---------- OCR preprocessing ----------

def preprocess_variants(crop_bgr):
    """Return two versions of the plate crop. OCR runs on both and the better reading wins."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(4, max(2, 120 // max(height, 1)))
    gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.bilateralFilter(enhanced, 5, 50, 50)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [smooth, binary]


# ---------- Plate parsing ----------

def map_chars(text, table):
    return "".join(table.get(char, char) for char in text)


def clean_ocr_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_regular(text):
    """Parse SS RR AA NNNN, for example MH 20 DV 2366."""
    if len(text) < 6:
        return None
    state = map_chars(text[:2], DIGIT_TO_LETTER)
    if state not in STATE_CODES:
        return None

    rest = text[2:]
    # Try a 2-digit RTO code first, then 1-digit, so "MH1DV2366" still parses.
    for rto_length in (2, 1):
        district = map_chars(rest[:rto_length], LETTER_TO_DIGIT)
        tail = rest[rto_length:]
        series_length = 0
        while series_length < min(3, len(tail)) and tail[series_length].isalpha():
            series_length += 1
        series = tail[:series_length]
        number = map_chars(tail[series_length:], LETTER_TO_DIGIT)

        if district.isdigit() and number.isdigit() and 1 <= len(number) <= 4:
            display = " ".join(part for part in (state, district, series, number) if part)
            return {
                "kind": "regular",
                "display": display,
                "state_code": state,
                "state_name": STATE_CODES[state],
                "fields": [
                    ("State / UT", f"{STATE_CODES[state]} ({state})"),
                    ("RTO code", district),
                    ("Series", series or "None"),
                    ("Number", number),
                ],
            }
    return None


def parse_bharat(text):
    """Parse YY BH NNNN L, for example 22 BH 1234 AA."""
    if len(text) not in (9, 10) or text[2:4] not in ("BH", "8H"):
        return None
    year = map_chars(text[:2], LETTER_TO_DIGIT)
    number = map_chars(text[4:8], LETTER_TO_DIGIT)
    suffix = text[8:]
    if not (year.isdigit() and number.isdigit() and suffix.isalpha()):
        return None
    return {
        "kind": "bharat",
        "display": f"{year} BH {number} {suffix}",
        "state_code": "BH",
        "state_name": "Bharat Series (valid across states)",
        "fields": [
            ("Series", "Bharat (BH), for vehicles that move between states"),
            ("Year code", year),
            ("Number", number),
            ("Suffix", suffix),
        ],
    }


def parse_plate(raw_text):
    cleaned = clean_ocr_text(raw_text)
    for prefix in EMBLEM_PREFIXES:
        if cleaned.startswith(prefix):
            candidate = cleaned[len(prefix):]
            parsed = parse_regular(candidate) or parse_bharat(candidate)
            if parsed:
                return parsed
    return None


# ---------- Detection and OCR ----------

def read_plate_text(reader, crop_bgr):
    best = {"raw": "", "ocr_confidence": 0.0, "parsed": None}
    best_score = (False, -1.0)

    for variant in preprocess_variants(crop_bgr):
        results = reader.readtext(variant, detail=1, allowlist=OCR_ALLOWLIST, paragraph=False)
        if not results:
            continue
        results.sort(key=lambda item: item[0][0][0])
        kept = [item for item in results if item[2] > 0.2] or results
        raw = "".join(item[1] for item in kept)
        confidence = float(np.mean([item[2] for item in kept]))
        parsed = parse_plate(raw)

        score = (parsed is not None, confidence)
        if score > best_score:
            best_score = score
            best = {"raw": raw, "ocr_confidence": confidence, "parsed": parsed}
    return best


def detect_and_read_plates(client, reader, frame_bgr, min_confidence=0.1):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    entry = run_workflow_with_retry(client, Image.fromarray(image_rgb))
    prediction_block = entry.get("predictions")
    if not prediction_block:
        return []

    raw_boxes = [box for box in prediction_block.get("predictions", []) if box["confidence"] >= min_confidence]
    frame_height, frame_width = frame_bgr.shape[:2]
    results = []
    for box in remove_duplicate_boxes(raw_boxes):
        x1 = max(0, int(box["x"] - box["width"] / 2) - 4)
        y1 = max(0, int(box["y"] - box["height"] / 2) - 4)
        x2 = min(frame_width, int(box["x"] + box["width"] / 2) + 4)
        y2 = min(frame_height, int(box["y"] + box["height"] / 2) + 4)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        ocr = read_plate_text(reader, crop)
        results.append({
            "box": (x1, y1, x2, y2),
            "detection_confidence": box["confidence"],
            **ocr,
        })
    return results


@st.cache_data(show_spinner=False)
def analyze_image(image_bytes):
    """Cached so moving the confidence slider doesn't re-run the models."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = detect_and_read_plates(get_client(), get_ocr_reader(), frame_bgr)
    return frame_bgr, results


# ---------- Display ----------

def draw_results(frame_bgr, results):
    annotated = frame_bgr.copy()
    for result in results:
        x1, y1, x2, y2 = result["box"]
        label = result["parsed"]["display"] if result["parsed"] else (result["raw"] or "?")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.7, 2)
        top = max(0, y1 - text_h - 14)
        cv2.rectangle(annotated, (x1, top), (x1 + text_w + 8, top + text_h + 10), (0, 200, 0), -1)
        cv2.putText(annotated, label, (x1 + 4, top + text_h + 4), FONT, 0.7, (0, 0, 0), 2)
    return annotated


def render_plate_card(number, result, show_raw):
    parsed = result["parsed"]
    st.markdown(f"#### Plate {number}")
    if parsed:
        st.metric("Registration number", parsed["display"])
        for label, value in parsed["fields"]:
            st.write(f"**{label}:** {value}")
        st.success("Matches the standard Indian plate format. This is a format check, not a registry lookup.")
    else:
        st.metric("Registration number", result["raw"] or "UNREADABLE")
        st.warning("Couldn't match the standard Indian plate format. Try a sharper, straighter photo.")

    st.caption(
        f"Detection {result['detection_confidence']:.2f} · OCR {result['ocr_confidence']:.2f}"
    )
    if show_raw:
        st.caption(f"Raw OCR text: {result['raw'] or '—'}")


def show_results(image_name, frame_bgr, results, threshold, show_raw):
    visible = [r for r in results if r["detection_confidence"] >= threshold]
    st.subheader(image_name)
    if not visible:
        st.info("No plates detected above the confidence threshold.")
        st.divider()
        return

    annotated_rgb = cv2.cvtColor(draw_results(frame_bgr, visible), cv2.COLOR_BGR2RGB)
    image_column, details_column = st.columns([3, 2])
    with image_column:
        st.image(annotated_rgb, use_container_width=True)
    with details_column:
        for number, result in enumerate(visible, 1):
            render_plate_card(number, result, show_raw)
            if number < len(visible):
                st.divider()
    st.divider()


st.set_page_config(page_title="License Plate Recognition", page_icon="🚗", layout="wide")
st.title("License Plate Recognition")
st.caption("Detect Indian number plates, read them, and show the registration details")

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Detection confidence threshold", 0.0, 1.0, 0.4, 0.05)
    show_raw = st.toggle("Show raw OCR text", value=False)
    st.divider()
    st.caption(f"Workspace: {ROBOFLOW_WORKSPACE}")

mode = st.radio("Choose input source", ["Upload image(s)", "Webcam snapshot"], horizontal=True)

if mode == "Upload image(s)":
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "jfif", "avif", "bmp", "webp"],
        accept_multiple_files=True,
    )
    for uploaded_file in uploaded_files or []:
        with st.spinner(f"Reading plates in {uploaded_file.name}..."):
            frame_bgr, results = analyze_image(uploaded_file.getvalue())
        show_results(uploaded_file.name, frame_bgr, results, conf_threshold, show_raw)
else:
    st.caption("Take a still photo with your browser webcam to analyze it.")
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        with st.spinner("Reading plates..."):
            frame_bgr, results = analyze_image(camera_image.getvalue())
        show_results("Webcam snapshot", frame_bgr, results, conf_threshold, show_raw)
