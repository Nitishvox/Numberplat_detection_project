# License Plate Recognition

Streamlit app that detects license plates with a Roboflow workflow and reads
plate text with EasyOCR. It supports multiple uploaded images and browser
webcam snapshots.

## Run locally

```powershell
python -m pip install -r requirements.txt
```

Edit `.streamlit/secrets.toml` and replace the placeholder API key, then run.
Use Python 3.11 or 3.12 because the Roboflow `inference-sdk` does not currently
provide a compatible package for Python 3.13:

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Streamlit Community Cloud

1. Push this folder to a GitHub repository. The real secrets file is ignored.
2. Set `app.py` as the app entry point.
3. Add the values from `.streamlit/secrets.toml.example` under the app's Secrets settings.

The webcam option captures a still photo from the browser, not continuous video.