import streamlit as st
import os
import time
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from Levenshtein import distance as levenshtein_distance
from dotenv import load_dotenv

load_dotenv()

AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

if not AZURE_KEY or not AZURE_ENDPOINT:
    st.error("Azure credentials missing.")
    st.stop()


@st.cache_resource
def get_azure_client():
    return ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))


def process_image_ocr(client, image_bytes):
    """Extracts text and bounding boxes from an image using Azure."""
    response = client.read_in_stream(BytesIO(image_bytes), raw=True)
    op_location = response.headers['Operation-Location']
    op_id = op_location.split('/')[-1]

    while True:
        result = client.get_read_result(op_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    extracted_data = []
    full_text = ""

    if result.status == OperationStatusCodes.succeeded:
        for page in result.analyze_result.read_results:
            for line in page.lines:
                full_text += line.text + " "
                extracted_data.append({"text": line.text, "box": line.bounding_box})

    return full_text.strip(), extracted_data


def draw_bounding_boxes(image_bytes, ocr_data):
    """Overlays bounding boxes on the original image."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for item in ocr_data:
        box = item['box']
        polygon = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
        draw.polygon(polygon, outline="red", width=3)

    return img


def calculate_metrics(ground_truth, predicted_text):
    """Calculates accuracy using Levenshtein distance."""
    if not ground_truth: return 0.0

    dist = levenshtein_distance(ground_truth.lower(), predicted_text.lower())
    max_len = max(len(ground_truth), len(predicted_text))

    accuracy = max(0, (max_len - dist) / max_len) * 100
    return round(accuracy, 2)


def apply_nlp_correction(raw_text, ground_truth, threshold=2):
    """Corrects misrecognized words based on the expected text lexicon."""
    lexicon = set(ground_truth.split())
    words = raw_text.split()
    corrected_words = []

    for word in words:
        best_match = word
        min_dist = float('inf')

        for valid_word in lexicon:
            dist = levenshtein_distance(word.lower(), valid_word.lower())
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                best_match = valid_word

        corrected_words.append(best_match)

    return " ".join(corrected_words)


def main():
    st.set_page_config(page_title="Handwriting OCR", layout="wide")
    st.title("🖋️ Smart OCR & NLP Correction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Input Data")
        uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        ground_truth = st.text_area("Ground Truth Text", height=100)

        use_nlp = st.checkbox("Apply NLP Lexicon Correction", value=False)

        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("2. AI Analysis")
        if uploaded_file and ground_truth:
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing..."):
                    client = get_azure_client()
                    image_bytes = uploaded_file.getvalue()

                    raw_text, ocr_data = process_image_ocr(client, image_bytes)
                    annotated_img = draw_bounding_boxes(image_bytes, ocr_data)

                    final_text = raw_text
                    if use_nlp:
                        final_text = apply_nlp_correction(raw_text, ground_truth)
                        st.info("💡 NLP Correction Applied")

                    st.image(annotated_img, caption="Text Localization", use_container_width=True)
                    st.success("Complete!")
                    st.code(final_text)

                    accuracy = calculate_metrics(ground_truth, final_text)
                    st.metric(label="Accuracy", value=f"{accuracy}%")


if __name__ == "__main__":
    main()