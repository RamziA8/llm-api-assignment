import streamlit as st
import easyocr
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
from groq import Groq
from PIL import Image, ImageDraw
import json
import tempfile

# load_dotenv("./.env")

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def run_ocr(image, page_num):
#     image_path = f"page_{page_num}.png"
#     image.save(image_path)
#     reader = easyocr.Reader(['en'])
#     results = reader.readtext(image_path, detail=1)
#     full_text = " ".join([item[1] for item in results])
#     return results, full_text

# st.title("PDF AI Assistant")
# st.write("Upload a PDF and ask questions about it!")

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file is not None:
#     st.success("PDF uploaded successfully!")

#     with st.spinner("Processing PDF with OCR... this may take a moment"):
#         # save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(uploaded_file.read())
#             tmp_path = tmp.name

#         # convert PDF to images
#         images = convert_from_path(tmp_path)

#         extracted_text = ""
#         ocr_results_per_page = []

#         for i, image in enumerate(images):
#             results, text = run_ocr(image, i + 1)
#             ocr_results_per_page.append(results)
#             extracted_text += text + "\n"

#     st.success("OCR complete!")

# all_chunks = []
# for page_idx, page_results in enumerate(ocr_results_per_page):
#     for i, (bbox, text, confidence) in enumerate(page_results):
#         all_chunks.append({
#             "id": f"{page_idx}_{i}",  # unique id like "0_5" = page 0, chunk 5
#             "text": text,
#             "page": page_idx,
#             "bbox": bbox
#         })

# system_message = f"""You are a helpful AI assistant. Answer questions based ONLY on this document.
# If the answer is not in the document, say 'This information is not in the document.'

# DOCUMENT:
# {extracted_text}"""

# # initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

#     # display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# user_input = st.chat_input("Ask a question about the document...")

# if user_input:
#         # show user message
#         with st.chat_message("user"):
#             st.write(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         response = client.chat.completions.create(
#             model=os.getenv("GROQ_MODEL"),
#             messages=[
#                 {"role": "system", "content": system_message}
#             ] + st.session_state.messages,
#             temperature=0.7,
#             max_tokens=500
#         )

#         ai_assistant_message = response.choices[0].message.content

#         with st.chat_message("assistant"):
#             st.write(ai_assistant_message)
#         st.session_state.messages.append({"role": "assistant", "content": ai_assistant_message})

load_dotenv("./.env")

# Must be the very first streamlit command called
st.set_page_config(layout="wide")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Cache EasyOCR Reader so it doesn't reload on every run
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

def run_ocr(image, page_num):
    image_path = f"page_{page_num}.png"
    image.save(image_path)
    reader = get_ocr_reader()
    results = reader.readtext(image_path, detail=1)
    full_text = " ".join([item[1] for item in results])
    return results, full_text

# Keep your fast search engine helper!
def get_relevant_chunks(question, all_chunks, max_chunks=20):
    question_words = set(question.lower().split())
    stop_words = {"what", "is", "the", "a", "an", "are", "i", "my", "for", "do", "does", "how", "much"}
    keywords = question_words - stop_words
    
    scored_chunks = []
    for chunk in all_chunks:
        chunk_text_lower = chunk["text"].lower()
        score = sum(1 for keyword in keywords if keyword in chunk_text_lower)
        if score > 0:
            scored_chunks.append((score, chunk))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:max_chunks]]

st.title("PDF AI Assistant with OCR Highlighting")
st.write("Upload your document, ask questions, and download the highlighted PDF!")

# --- SIDEBAR FOR FILE UPLOADS ---
with st.sidebar:
    st.header("Document Control")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    # Clear Chat button code:
    if st.session_state.messages:
        st.write("---")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.highlighted_pdf_path = None
            st.rerun()

# Initialize persistent session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = None
if "images" not in st.session_state:
    st.session_state.images = []
if "highlighted_pdf_path" not in st.session_state:
    st.session_state.highlighted_pdf_path = None

# --- PROCESS THE PDF (ONLY ONCE ON UPLOAD) ---
if uploaded_file is not None and st.session_state.all_chunks is None:
    with st.spinner("Processing PDF with OCR... this may take a moment"):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Convert and save images
        images = convert_from_path(tmp_path)
        st.session_state.images = images  # save to state

        extracted_text = ""
        ocr_results_per_page = []

        for i, image in enumerate(images):
            results, text = run_ocr(image, i + 1)
            ocr_results_per_page.append(results)
            extracted_text += text + "\n"

        # Build structured chunks
        all_chunks = []
        for page_idx, page_results in enumerate(ocr_results_per_page):
            for i, (bbox, text, confidence) in enumerate(page_results):
                all_chunks.append({
                    "id": f"{page_idx}_{i}",
                    "text": text,
                    "page": page_idx,
                    "bbox": bbox
                })
        
        st.session_state.all_chunks = all_chunks
        st.success("OCR completed! Ready to chat.")

# --- CONVERSATION & HIGHLIGHTING ---
if st.session_state.all_chunks is not None:
    
    # Display historical chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        # 1. User Message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        total_pages = len(st.session_state.images)
        if total_pages <= 5:
            relevant_chunks_for_answer = st.session_state.all_chunks
            st.toast(f"Short document ({total_pages} pgs). Sending full text to AI!")

        else:
        # 2. Grab only relevant chunks for context
            relevant_chunks_for_answer = get_relevant_chunks(user_input, st.session_state.all_chunks, max_chunks=20)
            st.toast(f"Long document ({total_pages} pgs). Using keyword filtering.")
        context = " ".join([c["text"] for c in relevant_chunks_for_answer])

        # Formulate a safe prompt
        system_message = f"""You are a helpful AI assistant. Answer questions based ONLY on the document context provided.
If the answer cannot be found in the context, politely state that you cannot find it.

Context from document: {context}"""

        # 3. Request Answer from Groq
        with st.spinner("Thinking..."):
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

        # 4. Ask Groq for Highlight Coordinates
        with st.spinner("Generating PDF Highlights..."):
            chunk_list = [{"id": c["id"], "text": c["text"]} for c in relevant_chunks_for_answer]
            
            reference_prompt = f"""
The user asked: "{user_input}"
Your answer: "{ai_assistant_message}"

Here are numbered text chunks from the document:
{chunk_list}

Return ONLY the IDs of chunks that directly support your answer to the question.
Return ONLY a JSON array of IDs, no explanation: ["0_5", "0_12"]
"""
            reference_response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL"),
                messages=[{"role": "user", "content": reference_prompt}],
                max_tokens=200
            )

            try:
                # Clean and parse the IDs
                raw_json = reference_response.choices[0].message.content.strip()
                relevant_ids = json.loads(raw_json)
                
                # Reload page images to clear old highlights
                for i, img in enumerate(st.session_state.images):
                    img.save(f"page_{i + 1}.png")

                # Highlight the matches
                for chunk in st.session_state.all_chunks:
                    if chunk["id"] in relevant_ids:
                        page_idx = chunk["page"]
                        bbox = chunk["bbox"]
                        image_path = f"page_{page_idx + 1}.png"

                        image = Image.open(image_path)
                        draw = ImageDraw.Draw(image, "RGBA")
                        
                        x1, y1 = bbox[0][0], bbox[0][1]
                        x2, y2 = bbox[2][0], bbox[2][1]
                        
                        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 0, 128))
                        image.save(image_path)

                # Save the new highlighted PDF
                highlighted_images = [
                    Image.open(f"page_{i + 1}.png").convert("RGB") 
                    for i in range(len(st.session_state.images))
                ]
                
                output_pdf_path = "highlighted_output.pdf"
                highlighted_images[0].save(
                    output_pdf_path,
                    save_all=True,
                    append_images=highlighted_images[1:]
                )
                
                st.session_state.highlighted_pdf_path = output_pdf_path
                st.rerun() # Refresh the page to show the download button

            except Exception as e:
                st.warning("Could not draw visual highlights for this answer.")

    # --- SHOW DOWNLOAD BUTTON IF HIGHLIGHTED PDF EXISTS ---
    if st.session_state.highlighted_pdf_path:
        with st.sidebar:
            st.write("---")
            st.success("Highlights updated!")
            with open(st.session_state.highlighted_pdf_path, "rb") as f:
                st.download_button(
                    label="Download Highlighted PDF",
                    data=f,
                    file_name="highlighted_policy.pdf",
                    mime="application/pdf"
                )