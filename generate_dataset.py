# generate_dataset.py
# Converts the Kaggle resume CSV into a JSONL file of prompt-response pairs for LLM fine-tuning.
# Each line in the output JSONL is a dict with 'prompt' and 'response' fields.
# Usage: python generate_dataset.py
# Output: data/my_dataset.jsonl
# See README for details.

from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import json
import openai
import pandas as pd
from tqdm import tqdm

# Set your OpenAI API key (or use environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please export your OpenAI API key before running this script.")

# Load the CSV (manually downloaded and unzipped)
csv_file = "./data/Resume/Resume.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found at {csv_file}. Please check your dataset extraction.")

# Load the CSV
df = pd.read_csv(csv_file)

# Helper: strip HTML tags
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Helper: truncate to ~1000 tokens (approx 4000 chars)
def truncate_text(text, max_chars=4000):
    return text[:max_chars]

# Prepare prompt/response pairs
examples = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    resume = clean_text(row.get('Resume_str', ''))
    category = str(row.get('Category', ''))
    resume = truncate_text(resume)
    if not resume or not category:
        continue
    prompt = f"You are an expert career coach. Given this resume for a {category} role:\n\n{resume}\n\nGenerate 3 strong resume bullet points in the appropriate tone and language."
    # Retry logic
    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            bullet_points = response.choices[0].message.content.strip()
            examples.append({"prompt": prompt, "response": bullet_points})
            break
        except Exception as e:
            print(f"API error: {e}. Retrying ({attempt+1}/3)...")
            time.sleep(2)
    time.sleep(1.5)

# Save as JSONL
out_path = "./data/my_dataset.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Saved {len(examples)} prompt/response pairs to {out_path}")
