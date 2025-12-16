import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np

# Caricamento e preparazione dei dati
df = pd.read_csv('./tweet_emotions.csv')
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

if 'user_input_processed' not in st.session_state:
    st.session_state['user_input_processed'] = None

# Funzione per il calcolo dei SHAP Values (viene chiamata su richiesta)
def calculate_shap_values(model,input_data):
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(input_data, check_additivity=False)
    return shap_values



# Funzione principale dell'app Streamlit
def main():
    st.title("Predizione delle Emozioni dai Testi")
    user_input = st.text_area("Inserisci il testo qui")
    prediction = None 

    if st.button("Predici"):
        user_input_processed = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(user_input_processed)
        st.write(f"Emozione predetta: {prediction[0]}")
        st.session_state['user_input_processed'] = user_input_processed
        st.session_state['prediction'] = prediction
    
    if st.button("Calcola SHAP Values"):
        if 'prediction' in st.session_state and st.session_state['prediction'] is not None:
                if 'user_input_processed' in st.session_state and st.session_state['user_input_processed'] is not None:
                    # Calcola i SHAP values per l'input dell'utente
                    shap_values = calculate_shap_values(model, st.session_state.user_input_processed)
                    
                    # Per semplicità, visualizza solo i valori SHAP per la classe predetta
                    predicted_class_index= list (model.classes_).index(st.session_state['prediction'][0])

                    #PROVA-------------------------------------------------------------------------
                    feature_names = vectorizer.get_feature_names_out()
                    input_words = user_input.lower().split()  # Assuming simple preprocessing
                    word_indices = [vectorizer.vocabulary_.get(word) for word in input_words if vectorizer.vocabulary_.get(word) is not None]

                    # Filter the SHAP values and input data for the corresponding words
                    shap_values_filtered = shap_values[predicted_class_index][:, word_indices]
                    user_input_processed_filtered = st.session_state['user_input_processed'][:, word_indices]
                    feature_names_filtered = np.array(feature_names)[word_indices]

                    # SICUROfeature_importance = np.abs(shap_values[predicted_class_index]).mean(axis=0)
                    # SICUROtop_features_indices = np.argsort(-feature_importance)[:len(user_input.split())]  # Prendi solo le prime N feature, dove N è il numero di parole nel testo dell'utente
                    # SICURO top_features_names = vectorizer.get_feature_names_out()[top_features_indices]

                    # Filtra i valori SHAP e i dati di input per le top features
                    #SICURO top_shap_values = shap_values[predicted_class_index][:, top_features_indices]
                    #SICURO top_user_input_processed = st.session_state['user_input_processed'][:, top_features_indices]


                    plt.figure()
                    shap.summary_plot(shap_values_filtered, features=user_input_processed_filtered, feature_names=feature_names_filtered)
                    # SICURO shap.summary_plot(top_shap_values, features= top_user_input_processed, feature_names = top_features_names)
                    st.pyplot(plt)
        else:
            st.write("Per favore, inserisci un testo e premi 'Predici' prima di calcolare i SHAP Values.")

if __name__ == "__main__":
    main()
