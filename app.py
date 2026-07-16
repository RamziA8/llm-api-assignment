import streamlit as st
import easyocr
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
from groq import Groq
from PIL import Image, ImageDraw
import json
import tempfile

load_dotenv("./.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_ocr(image, page_num):
    image_path = f"page_{page_num}.png"
    image.save(image_path)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path, detail=1)
    full_text = " ".join([item[1] for item in results])
    return results, full_text

st.title("PDF AI Assistant")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF with OCR... this may take a moment"):
        # save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # convert PDF to images
        images = convert_from_path(tmp_path)

        extracted_text = ""
        ocr_results_per_page = []

        for i, image in enumerate(images):
            results, text = run_ocr(image, i + 1)
            ocr_results_per_page.append(results)
            extracted_text += text + "\n"

    st.success("OCR complete!")

all_chunks = []
for page_idx, page_results in enumerate(ocr_results_per_page):
    for i, (bbox, text, confidence) in enumerate(page_results):
        all_chunks.append({
            "id": f"{page_idx}_{i}",  # unique id like "0_5" = page 0, chunk 5
            "text": text,
            "page": page_idx,
            "bbox": bbox
        })

system_message = f"""You are a helpful AI assistant. Answer questions based ONLY on this document.
If the answer is not in the document, say 'This information is not in the document.'

DOCUMENT:
{extracted_text}"""

# initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

    # display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about the document...")

if user_input:
        # show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=[
                {"role": "system", "content": system_message}
            ] + st.session_state.messages,
            temperature=0.7,
            max_tokens=500
        )

        ai_assistant_message = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(ai_assistant_message)
        st.session_state.messages.append({"role": "assistant", "content": ai_assistant_message})