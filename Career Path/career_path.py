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

'''
1.) Space Exploration (1-10)
"How interested are you in learning about space, planets, and stars?"
"Do you enjoy watching or reading about space missions and astronauts?"
2.) Scientific Experiments (1-10)
"How much do you enjoy doing science experiments or watching them?"
"Are you curious about how things work in nature?"
3.) Helping Others (1-10)
"How much do you like helping people when they're not feeling well?"
"Do you enjoy taking care of others?"
4.) Patience (1-10)
"How well can you wait for results when doing something?"
"Do you mind spending time on tasks that take a while?"
5.) Creativity (1-10)
"How much do you enjoy coming up with new ideas?"
"Do you like creating or designing things?"
6.) Empathy (1-10)
"How well do you understand how others might be feeling?"
"Do you care about how others are doing?"
Additional Information Questions:
7.) Favorite Subjects
"Which school subjects do you enjoy the most?"
"What topics do you like learning about?"
8.) Hobbies
"What activities do you enjoy doing in your free time?"
"What do you like to do for fun?" '''
# json_input = """
# {
#   "responses": {
#     "space_exploration": 10,
#     "scientific_experiments": 10,
#     "helping_others": 7.5,
#     "patience": 2.5,
#     "creativity": 10,
#     "empathy": 7.5
#   },
#   "additional_info": {
#     "favorite_subjects": ["Science", "Mathematics"],
#     "hobbies": ["Building models", "Reading about space"]
#   }
# }
# """

json_input = """
{
  "responses": {
    "space_exploration": 10,
    "scientific_experiments": 10,
    "helping_others": 7.5,
    "patience": 2.5,
    "creativity": 10,
    "empathy": 7.5
  },
  "additional_info": {
    "favorite_subjects": ["maths", "history"],
    "hobbies": ["reading", "music"]
  }
}
"""
student_data = json.loads(json_input)
assessment_df = pd.DataFrame([student_data])
print(assessment_df)    

def analyze_career_path(assessment_df):
    student_responses = assessment_df['responses'][0]  # Access the first row of 'responses' column
    query = (
        f"Analyze the career assessment responses for the student with ASD. "
        f"Based on their responses (scale 1-10):\n"
        f"Space Exploration: {student_responses['space_exploration']}\n"
        f"Scientific Experiments: {student_responses['scientific_experiments']}\n"
        f"Helping Others: {student_responses['helping_others']}\n"
        f"Patience: {student_responses['patience']}\n"
        f"Creativity: {student_responses['creativity']}\n"
        f"Empathy: {student_responses['empathy']}\n"
        f"Additional Information:\n"
        f"Favorite Subjects: {', '.join(assessment_df['additional_info'][0]['favorite_subjects'])}\n"
        f"Hobbies: {', '.join(assessment_df['additional_info'][0]['hobbies'])}\n"
        f"Recommend a Career Path out of the following options:(Astronaut/Scientist/Doctor) based on their responses and ASD considerations\n"
        f"ONLY respond with the recommended career path and only one. Do not add any explanation."
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

recommendation = analyze_career_path(assessment_df)
print(recommendation["choices"][0]["message"]["content"])
