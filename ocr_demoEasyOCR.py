from easyocr import Reader
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llm_client import call_llm
from pdf2image import convert_from_path
from groq import Groq

load_dotenv("./.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_ocr(image_path):
    # Check if the file actually exists first to avoid crashing
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    print("Initializing EasyOCR reader (this might take a second)...")
    # 'en' specifies English. You can add other language codes to this list.
    # If you don't have a dedicated Nvidia GPU, it will automatically fallback to CPU.
    reader = Reader(['en'])

    print(f"Extracting text from: {image_path}...\n")
    # detail=0 tells EasyOCR to only return a list of text strings
    results = reader.readtext(image_path, detail=0)

    full_text = " ".join(results)
    return full_text

def pdf_to_images(pdf_path):
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)
    return images

if __name__ == "__main__":
    PDF_FILE = "daman_schedule_of_benefits.pdf"

    images = pdf_to_images(PDF_FILE)

    extracted_text = ""

    # Run OCR on each page
    for i, image in enumerate(images):
        print(f"Processing page {i + 1}...")
        image_path = f"page_{i + 1}.png"
        image.save(image_path)
        text = run_ocr(image_path)
        extracted_text += text + "\n"

    if extracted_text:
        print("--- Extracted Text ---")
        print(extracted_text)

        prompt = f"Please provide a concise summary of the following text extracted from an image: {extracted_text}"

        summarized_text = call_llm(prompt, max_tokens=2000)
        print("--- Summarized Text ---")
        print(summarized_text)
    else:
        print("No text detected. Skipping LLM summary")
        print("----------------------")


system_message = "You are a helpful AI assistant. Answer questions about anything I ask clearly and concisely."

history_of_conversation = [{
    "role": "assistant", "content": f"I have analyzed the following document: {extracted_text}"
}]

print("AI Assistant (powered by Groq)")
print("Type 'quit' to exit\n")

name = input("Enter your name: ")
print(f"Assistant: Hello {name}! I'm your AI assistant. Ask me anything!\n")

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
    
    history_of_conversation.append({"role": "assistant", "content": ai_assistant_message})

    print(f"\nAssistant: {ai_assistant_message}\n")