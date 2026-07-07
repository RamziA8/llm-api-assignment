import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv("./.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_message = "You are a helpful football analyst assistant. Answer questions about football clearly and concisely."

# need something that allows me to write to AI on the terminal
# need something that remembers the previous messages

history_of_conversation = []

print("Football AI Assistant (powered by Groq)")
print("Type 'quit' to exit\n")

name = input("Enter your name: ")
print(f"Assistant: Hello {name}! I'm your football analyst assistant. Ask me anything about football!\n")

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
        {"role": "system", "content": "You are a helpful football analyst assistant."}
    ] + history_of_conversation,
    temperature=0.7,
    max_tokens=500
)
    ai_assistant_message = response.choices[0].message.content
    
    history_of_conversation.append({"role": "assistant", "content": ai_assistant_message})

    print(f"\nAssistant: {ai_assistant_message}\n")