import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llm_client import call_llm

load_dotenv("./.env")

client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

def ask_gemini_about_any_file(file_path, user_instruction):
    print(f"Uploading {file_path} directly to Gemini...")
    
    # This works whether file_path is "sample.png" OR "document.pdf"!
    uploaded_file = client.files.upload(file=file_path)
    
    print("Gemini is processing the file...")
    response = client.models.generate_content(
        model=os.getenv("LLM_MODEL"),
        contents=[uploaded_file, user_instruction]
    )
    
    return response.text

# Testing it in your main block:
if __name__ == "__main__":
    # You can change this to a PDF or a PNG, and it handles both perfectly!
    ANY_FILE = "daman_schedule_of_benefits.pdf" 
    
    extract_and_summary = ask_gemini_about_any_file(ANY_FILE, "First, extract and print all the text from this file, then please provide a summary of the file below.")
    print(extract_and_summary)