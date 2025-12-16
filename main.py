import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt



df = pd.read_csv("./tweet_emotions.csv")

print (df.head())

X = df['content']
Y = df['sentiment']  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)

vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)

def predict_emotion (text):

    text_trasformed = vectorizer.transform([text])

    prediction = model.predict(text_trasformed)

    background_sample = shap.sample(X_train_tfidf, 100)
    
    explainer = shap.KernelExplainer(model.predict, background_sample)

    #explainer = shap.Explainer(model.predict, X_train_tfidf)
    
    shap_values = explainer(text_trasformed)

    return prediction, shap_values

st.title ('Emotion Detection From Text')

user_input = st.text_input ('Inserisci il tuo testo qui: ')

if st.button ('Predire'):
    prediction, shap_values, text_trasformed = predict_emotion(user_input)
    st.success (f'Il tuo testo predice che il tuo sentimento sia {prediction[0]}')

    try:
        shap.plots.waterfall(shap_values[0], max_display= len (vectorizer.get_feature_names_out()), show = False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf() # Pulizia della figura
    except Exception as e:
        st.write(f"Si è verificato un errore durante la generazione del grafico SHAP: {e}")

    
    
    
    
    #fig,ax = plt.subplots()

    #shap.plots.waterfall(shap_values[0,0], show = False)
    
    #st.pyplot(fig)

    #plt.clf ()