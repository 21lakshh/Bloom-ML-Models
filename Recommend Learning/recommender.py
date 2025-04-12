import base64
import requests
import io
from dotenv import load_dotenv
import os
import logging
import json
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY is not set in the .env file")

json_input = """
{
  "topic": "Fractions",
  "learning_mode_used": "Gamified Learning",
  "quiz_results": false
}
"""

student_data = json.loads(json_input)
mistakes_df = pd.DataFrame([student_data], columns=['learning_mode_used', 'topic', 'quiz_results'])

print(mistakes_df)

def generate_recommendations(mistakes_df):
        
        query = (   
            f"You are an AI educational agent analyzing quiz results of a student learning {mistakes_df['topic'][0]} using {mistakes_df['learning_mode_used'][0]}. "
            f"The student got {mistakes_df['quiz_results'][0]} quiz result in this topic." 
            f"Based on the student's result, if the result is false,recommend a better learning mode else continue with the same learning mode."
            f"from the following: Storytelling, Music-Based Learning, or {mistakes_df['learning_mode_used'][0]}. "
            f"ONLY respond with the recommended learning mode. Do not add any explanation."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            }
        ]

        # API Request
        response = requests.post(
            GROQ_API_URL,
            json={"model": "llama-3.2-90b-vision-preview", "messages": messages, "max_tokens": 4000},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result

        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {"error": f"API Error: {response.status_code}"}
        
recommendations = generate_recommendations(mistakes_df)
print(recommendations["choices"][0]["message"]["content"])
