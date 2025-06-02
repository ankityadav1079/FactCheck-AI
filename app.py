# FakeCheck: AI Misinformation Hunter
# Full Hackathon Submission Code with Optional Features

import gradio as gr
from newspaper import Article
from transformers import pipeline
import pandas as pd
import uuid
import os

# Pipelines and models
classifier = pipeline("text-classification", model="microsoft/deberta-v3-base")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Claim log path
LOG_PATH = "claims_log.csv"

# Load or create log
if os.path.exists(LOG_PATH):
    claims_log = pd.read_csv(LOG_PATH)
else:
    claims_log = pd.DataFrame(columns=["claim", "verdict", "confidence"])

# Claim classifier logic
def classify_claim(claim):
    result = classifier(claim)[0]
    label = result['label']
    score = result['score']
    verdict = "True" if label == "LABEL_1" else "False"
    return verdict, score

# URL extraction
def extract_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return "Failed to extract text from the URL."

# Voice transcription
def transcribe(audio):
    return asr(audio)["text"]

# Confidence scoring
def confidence_score(score, sources=1):
    boost = 0.05 * min(sources, 5)
    return round(min(score + boost, 1.0), 2)

# Explanation generator
def explain_verdict(claim):
    prompt = f"Claim: {claim}\nExplain why it's true or false with reasoning."
    return generator(prompt, max_length=150)[0]['generated_text']

# Save claim
def save_claim(claim, verdict, score):
    global claims_log
    claims_log = pd.concat([
        claims_log,
        pd.DataFrame([[claim, verdict, score]], columns=["claim", "verdict", "confidence"])
    ], ignore_index=True)
    claims_log.to_csv(LOG_PATH, index=False)

# Leaderboard
def show_leaderboard():
    false_claims = claims_log[claims_log['verdict'] == "False"]
    return false_claims['claim'].value_counts().head(5).to_string()

# Main pipeline
def process_input(text, source_type):
    if source_type == "URL":
        text = extract_from_url(text)
    verdict, score = classify_claim(text)
    conf = confidence_score(score)
    explanation = explain_verdict(text)
    save_claim(text, verdict, conf)
    return text, verdict, f"{conf * 100:.1f}%", explanation

# Gradio Blocks UI
def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# 🔍 FakeCheck: AI Misinformation Hunter
        Enter a claim, paste a URL, or speak. FakeCheck will evaluate its truth, explain the reasoning, and provide confidence.
        """)

        with gr.Row():
            input_mode = gr.Radio(["Claim Text", "URL", "Voice"], label="Input Mode", value="Claim Text")

        claim_text = gr.Textbox(label="Claim or URL")
        audio_input = gr.Audio(source="microphone", type="filepath", label="Speak Now")
        submit = gr.Button("Verify Claim")

        with gr.Row():
            orig_claim = gr.Textbox(label="Processed Claim", interactive=False)
            verdict = gr.Textbox(label="Verdict", interactive=False)
            confidence = gr.Textbox(label="Confidence", interactive=False)

        explanation = gr.Textbox(label="Explanation", lines=4, interactive=False)
        leaderboard = gr.Textbox(label="🌟 Top Trending Fake Claims", lines=6, interactive=False)

        def route_input(mode, text, audio):
            if mode == "Voice":
                claim = transcribe(audio)
                return process_input(claim, "Claim")
            elif mode == "URL":
                return process_input(text, "URL")
            else:
                return process_input(text, "Claim")

        submit.click(route_input, [input_mode, claim_text, audio_input], [orig_claim, verdict, confidence, explanation])
        demo.load(show_leaderboard, None, leaderboard)

    return demo

# Launch UI
if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
