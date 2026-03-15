# Handwriting Recognition & NLP Correction

**Live Demo:** [Try the web app here](https://handwriting-ai-scanner-n29t4dxdtqfeznls4ew9nc.streamlit.app/)

**Video Demo:** [Watch how it works on YouTube](https://youtu.be/42DLJkKXEjg)

## About The Project
This is a web application that extracts text from images. It uses Azure Computer Vision for the initial OCR and a custom Natural Language Processing (NLP) pipeline for post-correction. 

To improve the raw AI output, the app uses the Levenshtein distance algorithm to compare the extracted text against an expected ground-truth lexicon. This automatically fixes common visual confusions (like mistaking an 'S' for an 'L') and improves the final recognition accuracy.

## Key Features
* **Azure OCR Integration:** Extracts handwritten text using the Azure Computer Vision API.
* **Text Localization:** Draws bounding boxes around detected words directly on the uploaded image.
* **NLP Auto-Correction:** Applies a Fuzzy String Matching algorithm (Levenshtein) to fix minor recognition mistakes based on the expected text.
* **Accuracy Metrics:** Calculates and displays the recognition accuracy percentage.
* **Web Interface:** Built entirely in Python using Streamlit.

## Built With
* **Language:** Python
* **Frontend:** Streamlit
* **Cloud API:** Azure Cognitive Services (Computer Vision)
* **Libraries:** Pillow, python-Levenshtein, python-dotenv
