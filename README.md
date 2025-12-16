# Modello di riconoscimento di emozioni dal testo 

Questo progetto utilizza diverse librerie e un modello pre-addestrato per l'emotion recognition. Sono state anche le Shap Values per una miglior comprensione di come il modello fornisce la sua predizione. Tramite l'utilizzo di plot caratteristici di Shap, possiamo comprendere come il modello seleziona le varie features per la predizione. Il modello utilizza le varie libreria CUDA per l'addestramento del modello, quindi assicurarsi di avere i requisiti adatti a livello hardware/driver

## Caratteristiche progetto
- Emotion recognition a partire da un testo inserito
- Visualizzazione delle predizioni e interpretazioni dei modelli.
- Creazione di un'applicazione web per interagire con il modello


## 🔎 SHAP Value (Shapley Additive Explanations)

Le **SHAP values** sono una tecnica di *Explainable AI (XAI)* utilizzata per interpretare le predizioni dei modelli di machine learning.
Si basano sulla **teoria dei giochi cooperativi**, in particolare sul **valore di Shapley**, e consentono di misurare **il contributo di ogni feature (o token) alla predizione finale del modello**. Nel contesto del *Natural Language Processing*, ogni **token/parola** di una frase può essere visto come un “giocatore” che contribuisce alla decisione del modello.



## 🧠 Cosa misura uno SHAP value

Per una singola predizione:

* uno **SHAP value positivo** indica che la feature/token **spinge la predizione verso una certa classe**
* uno **SHAP value negativo** indica che la feature/token **allontana la predizione da quella classe**
* il valore assoluto indica **l’intensità dell’influenza**

Le SHAP values soddisfano la proprietà di **additività**:

[
\text{Output del modello} = \text{valore base} + \sum_i \text{SHAP}_i
]

dove:

* **valore base** è la probabilità media della classe nel dataset
* **SHAP_i** è il contributo del singolo token


## 📌 Perché usare le SHAP values

Le SHAP values sono particolarmente utili perché:

* forniscono **spiegazioni locali**, interpretabili a livello di singola predizione
* sono **model-agnostic**
* permettono di analizzare il comportamento del modello su **testi reali**
* rendono interpretabili modelli complessi come **Transformer e reti neurali profonde**

## 📊 Esempio: predizione dell’emozione di una frase

Supponiamo un modello di **emotion classification** che assegna a una frase l’emozione predominante.

### Frase di input:

> *"Sono estremamente felice di aver raggiunto questo risultato!"*

### Classe predetta:

**Gioia** (probabilità 0.82)

### Valore base:

**0.35**
(probabilità media della classe *Gioia* nel dataset)

| Token        | SHAP value | Effetto sulla classe *Gioia* |
| ------------ | ---------- | ---------------------------- |
| estremamente | +0.12      | Rafforza l’intensità emotiva |
| felice       | +0.28      | Forte contributo positivo    |
| raggiunto    | +0.05      | Contesto positivo            |
| questo       | -0.01      | Contributo neutro            |
| risultato    | +0.03      | Associazione positiva        |

[
0.35 + 0.12 + 0.28 + 0.05 - 0.01 + 0.03 = 0.82
]

👉 Ogni SHAP value indica **quanto ciascun token ha contribuito alla predizione dell’emozione**.


## 🛠️ Esempio pratico (Python)

```python
import shap

# modello NLP già addestrato
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(["Sono estremamente felice di aver raggiunto questo risultato!"])

# visualizzazione token-level
shap.plots.text(shap_values)
```

La visualizzazione evidenzia i token **più rilevanti per la predizione dell’emozione**, mostrando in modo chiaro il contributo positivo o negativo di ciascuna parola.

## Come iniziare
Per utilizzare questo progetto, seguire i seguenti passi:
1. Clonare il repository.
2. Installare le dipendenze utilizzando `pip install -r requirements.txt`.
3. Avviare l'applicazione Streamlit con `streamlit run emotion_recognition.py`, o in alternativa `python -m streamlit run emotion_recognition.py`.
  3_1. In caso di problemi con keras, digitare da terminale: `pip install tf-keras --user`
  3_2. in caso di errori con la libreria transformers, disintallare e reinstallare:`pip uninstall transformers' e 'pip install transformers`

## Dipendenze
Questo progetto richiede le seguenti librerie, elencate nel file `requirements.txt`:

```
datasets
pandas
transformers
shap==0.44.1
streamlit
streamlit_shap
torch
matplotlib
```

## Autori
- de Stasio Giuseppe 
- Langiotti Andrea 
- Sergiacomi Daniele


## Contribuire
Se si desidera contribuire a questo progetto, si prega di fare un fork del repository e inviare una pull request.


