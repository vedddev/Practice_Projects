import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os
import streamlit as st

# Load Models
w2v_model=Word2Vec.load('word2vec_skipgram.model')
with open('sentiment_model.pkl','rb') as f:
    sentiment_model=pickle.load(f)
with open('label_mapping.pkl','rb') as f:
    label_mapping=pickle.load(f)

#reverse mapping

reverse_mapping={
    value:key for key,value in label_mapping.items()
}

#Avg_word2vec func
def avgword2vec(sentences,model):
    vector=[]
    for word in sentences:
        if word in model.wv:
            vector.append(model.wv[word])
    if len(sentences)==0:
        return np.zeros(model.vector_size)
    sentences_token=np.mean(vector,axis=0)
    return sentences_token


st.title("🎬 IMDb Movie Sentiment Analysis")

st.write("Enter a movie review and the model will predict its sentiment.")

review = st.text_area(
    "Enter your movie review:",
    height=150
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Tokenization
        tokens = word_tokenize(review.lower())

        # AvgWord2Vec
        review_vector = avgword2vec(tokens, w2v_model)

        # Reshape because model expects 2D input
        review_vector = review_vector.reshape(1, -1)

        # Prediction
        prediction = sentiment_model.predict(review_vector)[0]

        sentiment = reverse_mapping[prediction]

        # Display result
        if sentiment == "positive":
            st.success(" Positive Review")

        else:
            st.error(" Negative Review")