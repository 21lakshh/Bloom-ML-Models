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
  "student_name": "Anmol",
  "age": 10,
  "current_theme": "Astronaut",
  "total_questions_solved": 150,
  "total_modules_completed": 5,
  "current_topic": "Fractions",
  "learning_mode_used": "Gamified Learning",
  "quiz_results": true,
  "average_accuracy": 85,
  "favorite_subject": "Mathematics",
  "time_spent_learning": "2 hours daily",
  "strengths": ["Problem Solving", "Quick Learning"],
  "areas_for_improvement": ["Time Management", "Attention to Detail"]
}
"""

student_data = json.loads(json_input)
profile_df = pd.DataFrame([student_data], columns=['student_name', 'age', 'current_theme', 'total_questions_solved', 
                                                 'total_modules_completed', 'current_topic', 'learning_mode_used', 
                                                 'quiz_results', 'average_accuracy', 'favorite_subject', 
                                                 'time_spent_learning', 'strengths', 'areas_for_improvement'])

print(profile_df)

def generate_profile_summary(profile_df):
    query = (
        f"Generate a positive and encouraging profile summary for {profile_df['student_name'][0]}'s parents. "
        f"{profile_df['student_name'][0]} is {profile_df['age'][0]} years old and is currently exploring the {profile_df['current_theme'][0]} theme. "
        f"They have solved {profile_df['total_questions_solved'][0]} questions and completed {profile_df['total_modules_completed'][0]} modules. "
        f"In their current topic of {profile_df['current_topic'][0]}, they are using {profile_df['learning_mode_used'][0]} and have shown {profile_df['average_accuracy'][0]}% accuracy. "
        f"They spend {profile_df['time_spent_learning'][0]} on learning and particularly enjoy {profile_df['favorite_subject'][0]}. "
        f"Their key strengths include {', '.join(profile_df['strengths'][0])}. "
        f"Areas where they can grow include {', '.join(profile_df['areas_for_improvement'][0])}. "
        f"Please provide:\n"
        f"1. A positive summary of their progress\n"
        f"2. Specific achievements to celebrate\n"
        f"3. 2-3 constructive suggestions for parents to support their child's learning\n"
        f"4. 1-2 fun learning activities parents can do with their child\n"
        f"Keep the tone encouraging and focus on growth mindset."
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

recommendations = generate_profile_summary(profile_df)
print(recommendations["choices"][0]["message"]["content"])
