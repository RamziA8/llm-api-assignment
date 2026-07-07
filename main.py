import os
from dotenv import load_dotenv
from google import genai

load_dotenv("./.env")
print(os.getenv("LLM_MODEL"))
client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

response = client.models.generate_content(
    model=os.getenv("LLM_MODEL"),
    contents="Explain what a league table is in 3 simple sentences.",
)

print(response.text)