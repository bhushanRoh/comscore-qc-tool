# COMSCORE QC Tool

A Streamlit-based compliance quality control tool that scans promotional image boards for forbidden pharmaceutical brand names using OCR.

## Features

- **Multi-strategy OCR**: Uses 4 image preprocessing strategies + tile-based scanning for maximum text detection accuracy
- **Word-level highlighting**: Precisely highlights only the forbidden brand name within detected text, not surrounding words
- **Multi-panel board support**: Splits large boards into overlapping tiles for thorough scanning
- **Brand database management**: Add/remove forbidden brand names with comma-separated bulk input
- **Real-time compliance results**: Visual metrics showing violations found, unique brands, and text regions scanned

## Tech Stack

- **Streamlit** — Web interface
- **EasyOCR** — Text recognition engine
- **OpenCV** — Image preprocessing
- **Pandas** — Data management

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io) (free).
