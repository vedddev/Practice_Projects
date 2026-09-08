import os
import pickle
import string

import contractions
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from nltk.tokenize import word_tokenize


load_dotenv()


@st.cache_resource
def load_models():
    with open("ml_model.pkl", "rb") as f:
        ml_model = pickle.load(f)

    with open("sentiment_analysis.pkl", "rb") as f:
        w2v_model = pickle.load(f)

    return ml_model, w2v_model


@st.cache_resource
def load_groq():
    api_key = os.getenv("Groq_API_Key")

    if not api_key:
        return None

    return Groq(api_key=api_key)


LG_model, sentiment_model = load_models()
client = load_groq()


def preprocess(text):
    text = contractions.fix(text)
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]
    return tokens


def avgword2vec(tokens, model):
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def predict_sentiment(review):
    tokens = preprocess(review)

    review_vector = avgword2vec(
        tokens,
        sentiment_model
    ).reshape(1, -1)

    prediction = LG_model.predict(review_vector)[0]

    probabilities = dict(
        zip(
            LG_model.classes_,
            LG_model.predict_proba(review_vector)[0]
        )
    )

    return prediction, probabilities


def analyze(review, sentiment):
    if client is None:
        return "Groq API key was not found. Please check your .env file."

    prompt = f"""
You are an expert Amazon product review analyst.

Analyze this customer review independently.

Customer Review:
{review}

Machine Learning Prediction:
{sentiment}

The machine learning prediction may be incorrect. Determine the actual
sentiment from the review itself.

Provide the following:

### Summary
Give a short summary of the review.

### Positive Aspects
Mention the important positive points. If there are none, say "None".

### Negative Aspects
Mention the important complaints. If there are none, say "None".

### Customer Satisfaction
State whether the customer appears satisfied, neutral, or dissatisfied,
and briefly explain why.

### Recommendation
Give a short recommendation for a potential buyer.

Keep the response concise and easy to understand.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert product review analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


st.set_page_config(
    page_title="Amazon Review Analyzer",
    layout="centered"
)

st.title("Amazon Review Analyzer")

st.write(
    "Analyze customer reviews using Machine Learning and Generative AI."
)

review = st.text_area(
    "Enter your review",
    height=150,
    placeholder="Example: The product quality is excellent and I would definitely buy it again.",
    key="customer_review"
)

if st.button("Analyze Review", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        try:
            sentiment, probabilities = predict_sentiment(review)

            st.markdown("## Prediction")

            prediction_col, confidence_col = st.columns(2)

            with prediction_col:
                st.metric(
                    "Sentiment",
                    sentiment.title()
                )

            with confidence_col:
                confidence = probabilities[sentiment]
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

            st.markdown("### Sentiment Probabilities")

            for class_name, probability in probabilities.items():
                st.write(
                    f"**{class_name.title()}** — {probability:.2%}"
                )
                st.progress(float(probability))

            st.markdown("## AI Review Analysis")

            with st.spinner("Analyzing review with AI..."):
                result = analyze(
                    review,
                    sentiment
                )

            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #444;
                    background-color: rgba(128,128,128,0.08);
                ">
                {result.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")