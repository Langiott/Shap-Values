import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import shap
import streamlit as st

# Caricamento dei dati
df = pd.read_csv('./tweet_emotions.csv')

# Visualizzazione delle prime righe per capire i dati
print(df.head())

# Pre-elaborazione: Conversione del testo in vettori TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['sentiment']

# Divisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Puoi valutare il modello qui se lo desideri
predictions = model.predict(X_test)


# Inizializzazione dell'explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizzazione: per semplicità, visualizziamo solo il primo testo di esempio
shap.initjs()
grafico = shap.force_plot(explainer.expected_value[1], shap_values[1][0], feature_names=vectorizer.get_feature_names())



# Funzione principale dell'app Streamlit
def main():
    st.title("Predizione delle Emozioni dai Testi")
    user_input = st.text_area("Inserisci il testo qui")

    if st.button("Predici"):
        # Pre-elaborazione del testo dell'utente
        user_input_processed = vectorizer.transform([user_input]).toarray()
        
        # Predizione
        prediction = model.predict(user_input_processed)
        st.write(f"Emozione predetta: {prediction[0]}")

        # Calcolo SHAP Values (versione semplificata)
        # Qui potresti voler mostrare un grafico SHAP specifico per l'input dell'utente

        st.image(grafico, use_column_width=True)

st.set_page_config(page_title="App Predizione Emozioni", layout="wide")
if __name__ == "__main__":
    main()
