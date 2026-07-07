from llm_client import call_llm
from dotenv import load_dotenv
import time
import json

load_dotenv("./.env")

#prompt_a = "Write a 3-line dramatic match report for: Real Madrid 2 - 1 Barcelona. Make it sound like a sports commentator."
#prompt_b = "Write a 1-line dramatic match report for: Real Madrid 2 - 1 Barcelona. Make it sound like a sports commentator."

#print("=== EXPERIMENT A: Temperature ===\n")

temperatures = [0.0, 0.3, 0.7, 1.0]
#max_output_tokens = [30, 100, 600]

#for temp in temperatures:
    #print(f"--- Temperature: {temp} ---")
    #result = call_llm(prompt_a, temperature=temp)
    #print(result)
    #print()
    #time.sleep(3) # adding a small delay helps avoid hitting rate limits

#print("=== EXPERIMENT B: max_tokens ===\n")

#for token in max_output_tokens:
    #print(f"---Tokens: {token} ---")
    #result = call_llm(prompt_b, max_tokens=token)
    #print(result)
    #print() # print a blank line to add spacing between each temperature's output, making it easier to read.
    #time.sleep(3)

print("=== EXPERIMENT E: JSON reliability ===\n")
prompt_e = """ Extract the match data as JSON only.
Text: In a tense night in Madrid, Real Madrid beat Barcelona by two goals to one.
Required JSON keys:
{
"home_team": string,
"away_team": string,
"home_goals": integer,
"away_goals": integer,
"winner": string,
"is_draw": boolean
}"""

for temp in temperatures:
    print(f"--- Temperature: {temp} ---")
    result = call_llm(prompt_e, temperature = temp, max_tokens=1000)
    print(result)

    try:
        data = json.loads(result) # loads takes JSON string and converts it to a Python dictionary
        print("Valid JSON parsed successfully!")
        print(data)
    except json.JSONDecodeError: # invalid JSON string
        print("Failed to parse as JSON")
    print()
    time.sleep(3)