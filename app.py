import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["STREAMLIT_WATCH_FILES"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import streamlit as st
import pandas as pd
import spacy
import fitz  
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import time
import subprocess
import sys

try:
    import numpy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy  

nlp = spacy.load("en_core_web_sm")

try:
    import openpyxl
except ImportError:
    st.error("OpenPyXL is not installed. Install it using `pip install openpyxl`")

bert_model = SentenceTransformer("bert-base-nli-mean-tokens")
sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

st.title("AI-Powered Resume Screening & Sentiment Analysis")

st.subheader("Resume Screening")

resume_option = st.radio("Choose Resume Input Method", ("Single Resume", "Batch Processing (CSV/Excel)"))

if resume_option == "Single Resume":
    resume_input_option = st.radio("Resume Input Format", ("Paste Text", "Upload PDF"))

    if resume_input_option == "Paste Text":
        resume_text = st.text_area("Paste Resume Here")
    else:
        uploaded_file = st.file_uploader("Upload Resume (PDF from Any Location)", type=["pdf"])
        if uploaded_file:
            resume_text = extract_text_from_pdf(uploaded_file)  
            st.write("**Extracted Resume Text:**")
            st.text_area("", resume_text, height=200)

    job_description = st.text_area("Paste Job Description")

    if st.button("Check Resume Match"):
        if resume_text and job_description:
            resume_embedding = bert_model.encode(resume_text)
            job_embedding = bert_model.encode(job_description)
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100
            st.success(f"Resume Match Score: {similarity:.2f}%")
        else:
            st.warning("⚠ Please enter a resume and job description!")

if resume_option == "Batch Processing (CSV/Excel)":
    uploaded_file = st.file_uploader("Upload Resume File (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        required_columns = {"Name", "Resume_Text"}
        if not required_columns.issubset(df.columns):
            st.error(f"⚠ The uploaded file must contain the following columns: {', '.join(required_columns)}")
        else:
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())  

            job_description = st.text_area("Paste Job Description for Matching")

            if st.button("Process Resumes"):
                job_embedding = bert_model.encode(job_description)
                match_scores = []

                for index, row in df.iterrows():
                    resume_text = row["Resume_Text"]
                    resume_embedding = bert_model.encode(resume_text)
                    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100
                    match_scores.append((row["Name"], similarity_score))

                results_df = pd.DataFrame(match_scores, columns=["Name", "Match Score"])
                results_df = results_df.sort_values(by="Match Score", ascending=False)

                st.success("Resume Screening Completed!")
                st.dataframe(results_df)

                results_df.to_excel("resume_screening_results.xlsx", index=False, engine="openpyxl")
                with open("resume_screening_results.xlsx", "rb") as file:
                    st.download_button("Download Results", file, "resume_screening_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("Employee Sentiment Analysis")

sentiment_option = st.radio("Choose Sentiment Input Method", ("Single Feedback", "Batch Processing (CSV/Excel)"))

def get_engagement_strategy(sentiment_label):
    if sentiment_label == "POSITIVE":
        return "Keep up the good work! Recognize and reward employees to maintain high engagement."
    elif sentiment_label == "NEUTRAL":
        return "🔍 Conduct stay interviews to understand concerns before disengagement sets in."
    elif sentiment_label == "NEGATIVE":
        return " ⚠ Immediate intervention required! Schedule one-on-one meetings and address concerns to prevent attrition."
    else:
        return "Unable to determine sentiment. Please review the feedback manually."

if sentiment_option == "Single Feedback":
    feedback_text = st.text_area("Paste Employee Feedback")

    if st.button("Analyze Sentiment & Recommend Strategy"):
        if feedback_text:
            sentiment_result = sentiment_model(feedback_text)
            sentiment_label = sentiment_result[0]['label'].upper()
            confidence_score = sentiment_result[0]['score']
            
            engagement_strategy = get_engagement_strategy(sentiment_label)

            st.write(f"Sentiment: **{sentiment_label}**")
            st.write(f"Confidence: **{confidence_score:.2f}**")
            st.write(f"Recommended Strategy: **{engagement_strategy}**")
        else:
            st.warning("⚠ Please enter employee feedback!")

if sentiment_option == "Batch Processing (CSV/Excel)":
    uploaded_feedback_file = st.file_uploader("Upload Excel/CSV File (Employee Feedback)", type=["csv", "xlsx"])

    if uploaded_feedback_file:
        file_extension = uploaded_feedback_file.name.split(".")[-1]

        if file_extension == "csv":
            df_feedback = pd.read_csv(uploaded_feedback_file)
        elif file_extension == "xlsx":
            df_feedback = pd.read_excel(uploaded_feedback_file, engine="openpyxl")

        if "Feedback" not in df_feedback.columns:
            st.error("⚠ The uploaded file must contain a column named 'Feedback'.")
        else:
            st.write("Preview of Uploaded Feedback:")
            st.dataframe(df_feedback.head())

            if st.button("Process Feedback"):
                df_feedback["Sentiment"] = df_feedback["Feedback"].apply(lambda x: sentiment_model(x)[0]["label"].upper())
                df_feedback["Confidence"] = df_feedback["Feedback"].apply(lambda x: sentiment_model(x)[0]["score"])
                df_feedback["Engagement Strategy"] = df_feedback["Sentiment"].apply(get_engagement_strategy)

                st.success("Sentiment Analysis Completed!")
                st.dataframe(df_feedback)

                df_feedback.to_excel("sentiment_analysis_results.xlsx", index=False, engine="openpyxl")
                with open("sentiment_analysis_results.xlsx", "rb") as file:
                    st.download_button("Download Sentiment Results", file, "sentiment_analysis_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
