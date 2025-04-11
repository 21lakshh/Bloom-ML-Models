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
  "student_id": "STU_1043",
  "topic": "Fractions",
  "learning_mode_used": "Gamified Learning",
  "quiz_results": {
    "total_questions": 10,
    "correct_answers": 2,
    "mistakes": [
      {
        "question": "Which is greater: 1/2 or 3/4?",
        "student_answer": "1/2",
        "correct_answer": "3/4",
        "error_type": "comparison"
      },
      {
        "question": "Add 1/3 + 1/6",
        "student_answer": "2/9",
        "correct_answer": "1/2",
        "error_type": "common_denominator"
      }
    ]
  }
}
"""

student_data = json.loads(json_input)
mistakes_df = pd.DataFrame(student_data['quiz_results']['mistakes'], columns=['question', 'student_answer', 'correct_answer', 'error_type'])
mistakes_df['learning_mode_used'] = student_data['learning_mode_used']
mistakes_df['topic'] = student_data['topic']
mistakes_df['total_questions'] = student_data['quiz_results']['total_questions']
mistakes_df['correct_answers'] = student_data['quiz_results']['correct_answers']

print(mistakes_df)

def generate_recommendations(mistakes_df):
        
        query = (   
            f"You are an AI educational agent analyzing quiz results of a student learning {mistakes_df['topic'][0]} using {mistakes_df['learning_mode_used'][0]}. "
            f"The student got {mistakes_df['correct_answers'][0]} out of {mistakes_df['total_questions'][0]} questions correct. "
            f"Based on the student's mistakes, recommend a better learning mode "
            f"from the following: Storytelling, Music-Based Learning, or stick to {mistakes_df['learning_mode_used'][0]}. "
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
