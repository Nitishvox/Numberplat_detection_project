# License Plate Recognition

<p align="center">
	<strong>Detect. Read. Verify.</strong><br>
	A Streamlit computer-vision app for license-plate detection and OCR.
</p>

<p align="center">
	<a href="https://numberplatdetectionprojectgit-hbqkrawmgkdrsvnbjtnkg8.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white" alt="Live demo"></a>
	<a href="https://github.com/Nitishvox/Numberplat_detection_project"><img src="https://img.shields.io/badge/Source-GitHub-181717?logo=github&logoColor=white" alt="Source code"></a>
	<img src="https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white" alt="Python 3.12">
	<img src="https://img.shields.io/badge/License%20Plate%20OCR-EasyOCR-2ea44f" alt="EasyOCR">
</p>

## Overview

This application combines a Roboflow detection workflow with EasyOCR to find
license plates and read their text. Analyze one or more uploaded images or take
a still snapshot directly from a browser webcam.

## Features

- **Image upload:** Analyze JPG, PNG, BMP, WEBP, AVIF, and JFIF images.
- **Webcam capture:** Take a browser snapshot for quick inspection.
- **Roboflow detection:** Use a configurable hosted workflow for plate boxes.
- **OCR enhancement:** Upscale, normalize, and filter plate crops before OCR.
- **Duplicate filtering:** Remove overlapping detections with IoU suppression.
- **Confidence control:** Tune the detection threshold from the sidebar.
- **Result reporting:** View annotated images with detection and OCR confidence.

## Live Demo

Launch the deployed app: **[License Plate Recognition](https://numberplatdetectionprojectgit-hbqkrawmgkdrsvnbjtnkg8.streamlit.app/)**

The hosted app requires the Roboflow secret configured in Streamlit Community
Cloud. No API key is stored in this repository.

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/Nitishvox/Numberplat_detection_project.git
Set-Location Numberplat_detection_project
```

### 2. Create an environment

Use Python 3.12, matching the deployment runtime:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add
your real Roboflow API key:

```toml
ROBOFLOW_API_KEY = "your-roboflow-api-key"
ROBOFLOW_WORKSPACE = "m-nitish-46wkd"
ROBOFLOW_WORKFLOW_ID = "license-plate-recognition-vlicense-plate-recognition-rxg4e-l8dkk-1-yolo26n-t1-logic"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
```

The real `secrets.toml` file is ignored by Git. Never commit API keys.

### 4. Run the app

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deploy With Streamlit Community Cloud

1. Create a new app from `Nitishvox/Numberplat_detection_project`.
2. Select the `main` branch and set the entry point to `app.py`.
3. Open **Advanced settings**, then add the contents of the secrets template.
4. Replace the placeholder API key with your real Roboflow key.
5. Deploy or reboot the app after saving the secrets.

The repository includes `runtime.txt` to request Python 3.12 and pins a single
headless OpenCV package for the Linux deployment environment.

## Troubleshooting

| Message | Resolution |
| --- | --- |
| `ROBOFLOW_API_KEY is not set` | Add the key to local `.streamlit/secrets.toml` or Streamlit Cloud Secrets. |
| `ImportError` while importing `cv2` | Reboot the app and confirm it is using the `main` branch with the current `requirements.txt`. |
| No plates detected | Lower the confidence threshold and use an image with a clear, visible plate. |
| OCR is unreadable | Use a higher-resolution image with the plate facing the camera. |

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- runtime.txt
|-- .streamlit/
|   `-- secrets.toml.example
`-- README.md
```

## Technology Stack

![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?logo=opencv&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Recognition-2ea44f)
![Roboflow](https://img.shields.io/badge/Roboflow-Inference-111827)

## Security Note

API credentials belong in environment variables or Streamlit Secrets, never in
source code, screenshots, commit messages, or public issue reports.