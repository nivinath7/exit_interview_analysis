# pre_analyze.py

import openai
import json
from tqdm import tqdm # A library to show a cool progress bar!
import os

# --- SETUP ---
# You might need to install tqdm: pip install tqdm
# Set up your API key. For a script, we can use environment variables.
# Or, for simplicity in this demo, just paste it here (but don't share the file!).
# A better way is to load it from the .streamlit/secrets.toml file if you can.

openai.api_key = st.secrets["OPENAI_API_KEY"]


# --- AI ANALYSIS FUNCTION (Copied from app.py) ---
def analyze_interview(transcript_text):
    """
    Analyzes a single interview transcript using OpenAI's API.
    """
    prompt = f"""
    You are an expert HR analyst AI. Your task is to analyze an exit interview transcript and extract key insights.
    Analyze the following interview transcript and provide your output in a structured JSON format.

    Interview Transcript:
    ```
    {transcript_text}
    ```

    Instructions:
    Based on the transcript, generate a JSON object with the following exact schema:
    ```json
    {{
      "overallSentiment": "Positive | Negative | Neutral | Mixed",
      "keyThemes": ["<list of 1-3 most prominent themes from topics like Management, Compensation, etc.>"],
      "extractedEntities": [
        {{ "type": "Person", "name": "<e.g., Priya Sharma>" }},
        {{ "type": "Project", "name": "<e.g., Navigator Platform>" }}
      ],
      "summary": "<A 2-3 bullet point summary as a single string with newlines.>"
    }}
    ```
    CRITICAL: Only output the final JSON object. Do not include any other text or explanations.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

# --- MAIN SCRIPT LOGIC ---
def main():
    print("Loading original interview data...")
    with open('data.json', 'r') as f:
        interviews_data = json.load(f)

    analyzed_interviews = []
    print(f"Starting analysis of {len(interviews_data)} interviews...")

    # Use tqdm to create a nice progress bar in the terminal
    for interview in tqdm(interviews_data, desc="Analyzing Interviews"):
        transcript = interview.get("interviewTranscript", "")
        if transcript:
            analysis_result = analyze_interview(transcript)
            if analysis_result:
                # Combine original data with the new analysis
                combined_data = {**interview, "analysis": analysis_result}
                analyzed_interviews.append(combined_data)

    print("Analysis complete. Saving results to analyzed_data.json...")
    with open('analyzed_data.json', 'w') as f:
        json.dump(analyzed_interviews, f, indent=2)
    
    print("Done! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()
