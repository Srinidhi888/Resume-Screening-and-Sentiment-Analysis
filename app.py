import streamlit as st
import pandas as pd
import spacy
import fitz 
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os
import time
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


try:
    import openpyxl
except ImportError:
    st.error("OpenPyXL is not installed. Install it using `pip install openpyxl`")

nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("bert-base-nli-mean-tokens")
sentiment_model = pipeline("sentiment-analysis")

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file (works with any local file)."""
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
            resume_text = extract_text_from_pdf(uploaded_file)  # Extract text from uploaded PDF
            st.write("**Extracted Resume Text:**")
            st.text_area("", resume_text, height=200)

    job_description = st.text_area("Paste Job Description")

    if st.button("🔍 Check Resume Match"):
        if resume_text and job_description:
            resume_embedding = bert_model.encode(resume_text)
            job_embedding = bert_model.encode(job_description)
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100
            st.success(f"Resume Match Score: {similarity:.2f}%")
        else:
            st.warning("Please enter a resume and job description!")

if resume_option == "Batch Processing (CSV/Excel)":
    uploaded_file = st.file_uploader("Upload Resume File (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())  
        
        job_description = st.text_area("Paste Job Description for Matching")

        if st.button("Process Resumes"):
            if job_description:
                job_embedding = bert_model.encode(job_description)
                match_scores = []

                for index, row in df.iterrows():
                    if "Resume_Text" in df.columns:
                        resume_text = row["Resume_Text"]
                    else:
                        st.error("The uploaded file must contain a 'Resume_Text' column.")
                        break
                    
                    doc = nlp(resume_text)
                    skills = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                    
                    resume_embedding = bert_model.encode(resume_text)
                    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100
                    
                    match_scores.append((row["Name"], similarity_score, skills))
                
                results_df = pd.DataFrame(match_scores, columns=["Name", "Match Score", "Skills"])
                results_df = results_df.sort_values(by="Match Score", ascending=False)
                
                st.success("Resume Screening Completed!")
                st.dataframe(results_df)

                results_df.to_excel("resume_screening_results.xlsx", index=False, engine="openpyxl")

                with open("resume_screening_results.xlsx", "rb") as file:
                    st.download_button("📥 Download Results", file, "resume_screening_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("Employee Sentiment Analysis")

sentiment_option = st.radio("Choose Sentiment Input Method", ("Single Feedback", "Batch Processing (CSV/Excel)"))

if sentiment_option == "Single Feedback":
    feedback_text = st.text_area("Paste Employee Feedback")

    if st.button("Analyze Sentiment"):
        if feedback_text:
            sentiment_result = sentiment_model(feedback_text)
            st.write(f"🎭 Sentiment: **{sentiment_result[0]['label']}**")
            st.write(f"Confidence: **{sentiment_result[0]['score']:.2f}**")
        else:
            st.warning("Please enter employee feedback!")

if sentiment_option == "Batch Processing (CSV/Excel)":
    uploaded_feedback_file = st.file_uploader("Upload Excel/CSV File (Employee Feedback)", type=["csv", "xlsx"])

    if uploaded_feedback_file:
        file_extension = uploaded_feedback_file.name.split(".")[-1]

        if file_extension == "csv":
            df_feedback = pd.read_csv(uploaded_feedback_file)
        elif file_extension == "xlsx":
            df_feedback = pd.read_excel(uploaded_feedback_file, engine="openpyxl")

        st.write("Preview of Uploaded Feedback:")
        st.dataframe(df_feedback.head())

        if st.button("Process Feedback"):
            if "Feedback" not in df_feedback.columns:
                st.error("The uploaded file must have a column named 'Feedback'")
            else:
                batch_size = 20  
                total_feedbacks = len(df_feedback)
                processed_feedbacks = []

                for i in range(0, total_feedbacks, batch_size):
                    batch = df_feedback["Feedback"].iloc[i : i + batch_size].tolist()
                    results = sentiment_model(batch)

                    for j, result in enumerate(results):
                        processed_feedbacks.append(
                            (
                                df_feedback.iloc[i + j]["Feedback"],
                                result["label"],
                                result["score"],
                            )
                        )

                    progress = (i + batch_size) / total_feedbacks
                    st.progress(min(progress, 1.0))
                    time.sleep(0.5)

                sentiment_df = pd.DataFrame(processed_feedbacks, columns=["Feedback", "Sentiment", "Confidence"])
                st.success("Sentiment Analysis Completed!")
                st.dataframe(sentiment_df)

                sentiment_df.to_excel("sentiment_analysis_results.xlsx", index=False, engine="openpyxl")

                with open("sentiment_analysis_results.xlsx", "rb") as file:
                    st.download_button("Download Sentiment Results", file, "sentiment_analysis_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
