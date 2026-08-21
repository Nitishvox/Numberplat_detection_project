import logging
import os
import time

import cv2
import easyocr
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image


def get_secret(key, default=None):
    if key in st.secrets:
        return st.secrets[key]
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
            logger.warning(
                "Workflow call failed (attempt %s/%s): %s. Retrying in %.1fs",
                attempt + 1,
                retries + 1,
                error,
                wait,
            )
            time.sleep(wait)
    raise PlateDetectionError(
        f"Workflow call failed after {retries + 1} attempts: {last_error}"
    )


def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    def iou(first, second):
        first_x1 = first["x"] - first["width"] / 2
        first_y1 = first["y"] - first["height"] / 2
        first_x2 = first["x"] + first["width"] / 2
        first_y2 = first["y"] + first["height"] / 2
        second_x1 = second["x"] - second["width"] / 2
        second_y1 = second["y"] - second["height"] / 2
        second_x2 = second["x"] + second["width"] / 2
        second_y2 = second["y"] + second["height"] / 2
        intersection = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
        intersection *= max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
        first_area = first["width"] * first["height"]
        second_area = second["width"] * second["height"]
        return intersection / (first_area + second_area - intersection + 1e-6)

    sorted_boxes = sorted(boxes, key=lambda box: -box["confidence"])
    kept = []
    for box in sorted_boxes:
        if all(iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


def preprocess_plate_crop(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = max(1, 300 // max(height, 1))
    gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.bilateralFilter(gray, 5, 50, 50)


def read_plate_text(reader, crop_bgr):
    ocr_result = reader.readtext(
        preprocess_plate_crop(crop_bgr),
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        paragraph=False,
    )
    ocr_result.sort(key=lambda result: result[0][0][0])
    text = "".join(result[1] for result in ocr_result if result[2] > 0.2).strip()
    average_confidence = float(np.mean([result[2] for result in ocr_result])) if ocr_result else 0.0
    return text, average_confidence


def detect_and_read_plates(client, reader, frame_bgr, conf_threshold=0.4):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    entry = run_workflow_with_retry(client, Image.fromarray(image_rgb))
    prediction_block = entry.get("predictions")
    if not prediction_block:
        return []

    raw_boxes = prediction_block.get("predictions", [])
    boxes = remove_duplicate_boxes(
        [box for box in raw_boxes if box["confidence"] >= conf_threshold]
    )
    frame_height, frame_width = frame_bgr.shape[:2]
    results = []
    for box in boxes:
        x1 = max(0, int(box["x"] - box["width"] / 2) - 4)
        y1 = max(0, int(box["y"] - box["height"] / 2) - 4)
        x2 = min(frame_width, int(box["x"] + box["width"] / 2) + 4)
        y2 = min(frame_height, int(box["y"] + box["height"] / 2) + 4)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        plate_text, ocr_confidence = read_plate_text(reader, crop)
        results.append({
            "box": (x1, y1, x2, y2),
            "detection_confidence": box["confidence"],
            "plate_text": plate_text,
            "ocr_confidence": ocr_confidence,
        })
    return results


def draw_results(frame_bgr, results):
    annotated = frame_bgr.copy()
    for result in results:
        x1, y1, x2, y2 = result["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            result["plate_text"] or "?",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return annotated


def show_results(image_name, frame_bgr, results):
    annotated_rgb = cv2.cvtColor(draw_results(frame_bgr, results), cv2.COLOR_BGR2RGB)
    st.subheader(image_name)
    result_column, details_column = st.columns([2, 1])
    with result_column:
        st.image(annotated_rgb, use_container_width=True)
    with details_column:
        if results:
            for result in results:
                st.metric("Plate", result["plate_text"] or "UNREADABLE")
                st.caption(
                    f"Detection: {result['detection_confidence']:.2f} | "
                    f"OCR: {result['ocr_confidence']:.2f}"
                )
        else:
            st.info("No plates detected.")


st.set_page_config(page_title="License Plate Recognition", page_icon="car", layout="wide")
st.title("License Plate Recognition")
st.caption("Detect and read license plates using your Roboflow workflow")

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Detection confidence threshold", 0.0, 1.0, 0.4, 0.05)
    st.divider()
    st.caption(f"Workspace: {ROBOFLOW_WORKSPACE}")
    st.caption(f"Workflow: {ROBOFLOW_WORKFLOW_ID[:40]}...")

mode = st.radio("Choose input source", ["Upload image(s)", "Webcam snapshot"], horizontal=True)
client = get_client()
reader = get_ocr_reader()

if mode == "Upload image(s)":
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "jfif", "avif", "bmp", "webp"],
        accept_multiple_files=True,
    )
    for uploaded_file in uploaded_files or []:
        image = Image.open(uploaded_file).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner(f"Detecting plates in {uploaded_file.name}..."):
            results = detect_and_read_plates(client, reader, frame_bgr, conf_threshold)
        show_results(uploaded_file.name, frame_bgr, results)
        st.divider()
else:
    st.caption("Take a still photo with your browser webcam to analyze it.")
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting plates..."):
            results = detect_and_read_plates(client, reader, frame_bgr, conf_threshold)
        show_results("Webcam snapshot", frame_bgr, results)