import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv("./.env")

client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

def call_llm(prompt, system_message="You are a helpful assistant.", temperature=0.3, max_tokens=250):
    try:
        response = client.models.generate_content(
            model=os.getenv("LLM_MODEL"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text
    except Exception as error:
        return f"API call failed: {error}"