import easyocr
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llm_client import call_llm
from pdf2image import convert_from_path
from groq import Groq
import fitz  # pymupdf
import json
from PIL import Image, ImageDraw

# load_dotenv("./.env")

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def run_ocr(image_path):
#     # Check if the file actually exists first to avoid crashing
#     if not os.path.exists(image_path):
#         print(f"Error: The file '{image_path}' was not found.")
#         return

#     print("Initializing EasyOCR reader (this might take a second)...")
#     # 'en' specifies English. You can add other language codes to this list.
#     # If you don't have a dedicated Nvidia GPU, it will automatically fallback to CPU.
#     reader = easyocr.Reader(['en'])

#     print(f"Extracting text from: {image_path}...\n")
#     # detail=0 tells EasyOCR to only return a list of text strings
#     results = reader.readtext(image_path, detail=1)
#     # print(results)
#     full_text = " ".join(item[1] for item in results)
#     # return results, full_text
#     return full_text

# def pdf_to_images(pdf_path):
#     print("Converting PDF to images...")
#     images = convert_from_path(pdf_path)
#     return images

# if __name__ == "__main__":
#     PDF_FILE = "daman_schedule_of_benefits.pdf"

#     images = pdf_to_images(PDF_FILE)

#     extracted_text = ""

#     # Run OCR on each page
#     for i, image in enumerate(images):
#         print(f"Processing page {i + 1}...")
#         image_path = f"page_{i + 1}.png"
#         image.save(image_path)
#         text = run_ocr(image_path)
#         extracted_text += text + "\n"

#     if extracted_text:
#         print("--- Extracted Text ---")
#         print(extracted_text)

#         prompt = f"Please provide a concise summary of the following text extracted from an image: {extracted_text}"

#         summarized_text = call_llm(prompt, max_tokens=2000)
#         print("--- Summarized Text ---")
#         print(summarized_text)
#     else:
#         print("No text detected. Skipping LLM summary")
#         print("----------------------")


# system_message = "You are a helpful AI assistant. Answer questions about anything I ask clearly and concisely."

# history_of_conversation = [{
#     "role": "assistant", "content": f"I have analyzed the following document: {extracted_text}"
# }]

# print("AI Assistant (powered by Groq)")
# print("Type 'quit' to exit\n")

# name = input("Enter your name: ")
# print(f"Assistant: Hello {name}! I'm your AI assistant. Ask me anything!\n")

# while True:
#     input_text = input("You: ")

#     if input_text == "quit":
#         print("Conversation ended. Goodbye!")
#         break

#     history_of_conversation.append({
#         "role": "user",
#         "content": input_text
#     })

#     response = client.chat.completions.create(
#     model=os.getenv("GROQ_MODEL"),
#     messages=[
#         {"role": "system", "content": system_message}
#     ] + history_of_conversation,
#     temperature=0.7,
#     max_tokens=500
# )
#     ai_assistant_message = response.choices[0].message.content
    
#     history_of_conversation.append({"role": "assistant", "content": ai_assistant_message})

#     print(f"\nAssistant: {ai_assistant_message}\n")


# def highlight_pdf(pdf_path, phrases_to_highlight, output_path):
#     doc = fitz.open(pdf_path)

#     for page in doc:
#         for phrase in phrases_to_highlight:
#             for instance in page.search_for(phrase):
#                 page.add_highlight_annot(instance)

#     doc.save(output_path)
#     print(f"Highlighted pdf saved as {output_path}")

# reference_prompt = f"""
# From this document: {extracted_text}
# You just answered this question that the user asked: {input_text}
# You just answered: {ai_assistant_message}
# Find the EXACT sentences or paragraphs from the document that support the answer.
# Return as JSON list only: ["phrase1", "phrase2"]
# """

# reference_response = client.chat.completions.create(
#     model=os.getenv("GROQ_MODEL"),
#     messages=[{"role": "user", "content": reference_prompt}],
#     max_tokens=1500
# )

# import json
# phrases = json.loads(reference_response.choices[0].message.content)
# highlight_pdf("daman_schedule_of_benefits.pdf", phrases, "highlighted_output.pdf")


load_dotenv("./.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_ocr(image_path):  
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return [], ""

    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en']) 
    print(f"Extracting text from: {image_path}...\n")
    results = reader.readtext(image_path, detail=1)
    full_text = " ".join([item[1] for item in results])
    return results, full_text

def pdf_to_images(pdf_path):
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)
    return images

if __name__ == "__main__":
    PDF_FILE = "daman_schedule_of_benefits.pdf"

    images = pdf_to_images(PDF_FILE)

    extracted_text = ""
    ocr_results_per_page = []
    
    for i, image in enumerate(images):
        print(f"Processing page {i + 1}...")
        image_path = f"page_{i + 1}.png"
        image.save(image_path)
        results, text = run_ocr(image_path)
        ocr_results_per_page.append(results)
        extracted_text += text + "\n"

    if extracted_text:
        print("--- Extracted Text ---")
        print(extracted_text)

        prompt = f"Please provide a concise summary of the following text: {extracted_text}"
        summarized_text = call_llm(prompt, max_tokens=2000)
        print("--- Summarized Text ---")
        print(summarized_text)
    else:
        print("No text detected.")

    # need to create a list of dictionaries, all_chunks, where each dictionary represents one text block found by OCR

    # -------------------------
    # BUILD STRUCTURED CHUNKS WITH IDs
    # -------------------------
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

    history_of_conversation = []

    print("\nAI Assistant (powered by Groq)")
    print("Type 'quit' to exit\n")

    name = input("Enter your name: ")
    print(f"Assistant: Hello {name}! I have analyzed the document. Ask me anything about it!\n")

    while True:
        input_text = input("You: ")

        if input_text == "quit":
            print("Conversation ended. Goodbye!")
            break

        history_of_conversation.append({
            "role": "user",
            "content": input_text
        })

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=[
                {"role": "system", "content": system_message}
            ] + history_of_conversation,
            temperature=0.7,
            max_tokens=500
        )

        ai_assistant_message = response.choices[0].message.content

        history_of_conversation.append({
            "role": "assistant",
            "content": ai_assistant_message
        })

        print(f"\nAssistant: {ai_assistant_message}\n")


        # send only id and text to Groq, not bbox because it doesnt know what the image looks like, only understands text
        chunk_list = [{"id": c["id"], "text": c["text"]} for c in all_chunks]

        reference_prompt = f"""
The user asked: "{input_text}"
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
            relevant_ids = json.loads(reference_response.choices[0].message.content)
            print(f"Relevant chunk IDs: {relevant_ids}")

            # reload original page images to highlight fresh each time
            # for i in range(len(images)):
            #     images[i].save(f"page_{i + 1}.png")

            # find matching chunks and draw highlights
            for chunk in all_chunks:
                if chunk["id"] in relevant_ids:
                    page_idx = chunk["page"]
                    bbox = chunk["bbox"]
                    image_path = f"page_{page_idx + 1}.png"

                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image, "RGBA") # red, green, blue, alpha. alpha is the transparency

#                     bbox = [
#     [x1, y1],  # corner 0 - top left
#     [x2, y1],  # corner 1 - top right
#     [x2, y2],  # corner 2 - bottom right
#     [x1, y2]   # corner 3 - bottom left
# ]
                    # e.g. of how it works
                    # bbox[0]: go to corner 0 (top left), gives you [x1, y1]
                    # bbox[0][0]: go to first number in that corner, gives you x1
                    # bbox[0]: corner 0 (top left), [x1, y1]
                    # bbox[0][1]: second number, gives you y1

                    x1 = bbox[0][0]
                    y1 = bbox[0][1]
                    x2 = bbox[2][0]
                    y2 = bbox[2][1]
                    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 0, 128))
                    image.save(image_path)

            # convert highlighted images back to one PDF
            highlighted_images = []
            for i in range(len(images)):
                highlighted_images.append(
                    Image.open(f"page_{i + 1}.png").convert("RGB") # must convert from RGBA to RGB because pdf doesnt understand transparency, so there would be an error
                )

            highlighted_images[0].save(
                "highlighted_output.pdf",
                save_all=True,
                append_images=highlighted_images[1:]
            )
            print("Highlighted PDF saved as highlighted_output.pdf\n")

        except json.JSONDecodeError:
            print("Could not generate highlighted reference.\n")