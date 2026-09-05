from multiprocessing.connection import Client
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


with open('sentiment_model.pkl','rb') as f:
    sentiment_model=pickle.load(f)
with open('tfidf_vectorizer.pkl','rb') as f:
    tfidf_model=pickle.load(f)


groq_api_key=os.getenv('Groq_API_Key')
client=Groq(api_key=groq_api_key)




st.title('Review System')
review=st.text_area(
    "Enter your Review:",
    height=150
)

# print("API key loaded:", groq_api_key is not None)
# models = client.models.list()

# for model in models.data:
#     print(model.id)
def analyze(review,sentiment):

    prompt = f"""
You are a product review analyst.

Analyze the following Redmi smartphone customer review.

Customer Review:
{review}

Machine Learning Sentiment:
{sentiment}

IMPORTANT:
The machine learning prediction may be incorrect.
Analyze the actual customer review independently.
Do not blindly trust the ML prediction.

If the review clearly expresses negative sentiment but the ML model
predicts positive, identify the review as negative.

Provide:

1. A short summary.
2. Key positive aspects.
3. Key complaints or negative aspects.
4. Overall customer satisfaction.
5. A short recommendation for a potential buyer.

Keep the response concise and easy to understand.
"""
    response=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role":"system",
                "content":"You are an expert product review analyst"
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

if st.button("Analyze Review"):
    if  review.strip()=="":
        st.warning('Enter the review bro')
    else:
        tokens=word_tokenize(review.lower())
        clean_tokens=[
            word for word in tokens if word.isalnum()
        ]

        clean_review=" ".join(clean_tokens)
        review_tfidf=tfidf_model.transform([clean_review])
        prediction=sentiment_model.predict(review_tfidf)[0]
        st.subheader('Sentiment')

        if prediction=="positive":
            st.success("Positive")
        elif prediction=="negative":
            st.error("Negative")
        else:
            st.warning("Neutral")

        st.subheader('Review analysis')
        with st.spinner("Analyzing review with AI...."):
            analysis=analyze(
                review,
                prediction
            )
        st.write(analysis)

# st.write("ML Prediction:", prediction)