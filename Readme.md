# Asistent de Diagnostic Medical (AI-Powered)

Acest proiect reprezintă un sistem inteligent de asistență pentru diagnosticul medical, utilizând tehnici avansate de Procesare a Limbajului Natural (NLP) și Machine Learning. Aplicația analizează simptomele introduse de utilizator sub formă de text liber și sugerează cele mai probabile afecțiuni prin compararea cu o bază de date extinsă.

---

## Funcționalități Principale

* **Procesare NLP Multilingvă:** Utilizează `Sentence Transformers` pentru a înțelege contextul simptomelor, nu doar cuvintele cheie.
* **Algoritm de Căutare Eficient:** Implementează `K-Nearest Neighbors (KNN)` cu metrică de similaritate cosinus pentru rezultate precise.
* **Optimizare prin PCA:** Reduce dimensionalitatea datelor pentru o viteză de răspuns instantanee, fără a pierde din acuratețe.
* **Sistem de Cache Inteligent:** Salvează modelele antrenate local pentru a evita recalcularea lor la fiecare pornire.
* **Interfață Desktop Intuitivă:** Chatbot integrat realizat cu `Gradio` și rulat într-o fereastră nativă de sistem prin `PyWebView`.
* **Mod Offline:** Mecanism de fallback pe `TF-IDF` în cazul lipsei conexiunii la internet pentru descărcarea modelelor transformer.

---

## Instalare și Cerințe

Pentru a rula acest proiect, aveți nevoie de Python 3.8+ și următoarele biblioteci:

```bash
pip install pandas numpy scikit-learn sentence-transformers joblib gradio pywebview
```

## Utilizare

Pentru a lansa aplicația, rulați scriptul principal:

```bash
python nume_fisier.py
```

### Fluxul de Lucru

**1. Prima Rulare (Inițializare):**
* Sistemul caută setul de date: `Final_Augmented_dataset_Diseases_and_Symptoms.csv`.
* Se generează embedding-urile (vectorii) pentru simptome.
* Se aplică transformarea PCA și se construiește indexul de căutare.
* Toate aceste componente sunt salvate în fișiere de tip cache.

**2. Rulările Ulterioare (Rapidă):**
* Sistemul detectează automat fișierele `.pkl` și `.npy`.
* Modelele sunt încărcate instantaneu, iar aplicația pornește în câteva secunde.

---

## Structura Fișierelor de Cache

După prima rulare, vor apărea următoarele fișiere în directorul proiectului:

| Fișier | Descriere |
| :--- | :--- |
| `medical_model_dataframe.pkl` | Baza de date a bolilor optimizată pentru acces rapid. |
| `medical_model_embeddings.npy` | Matricea vectorială pre-calculată a simptomelor. |
| `medical_model_pca.pkl` | Modelul antrenat pentru reducerea dimensiunilor. |
| `medical_model_index.pkl` | Indexul KNN gata de utilizare pentru căutare. |
| `cached_symptom_data.pkl` | Cache pentru maparea rapidă a simptomelor. |

---

## Disclaimer Medical (Atenție!)

**IMPORTANT:** Acest software este un experiment educațional și de cercetare în domeniul inteligenței artificiale.

* **NU** reprezintă un dispozitiv medical certificat.
* **NU** înlocuiește consultul, diagnosticul sau tratamentul oferit de un medic profesionist.
* Rezultatele oferite sunt bazate pe probabilități statistice și pot fi eronate.
* Consultați întotdeauna un cadru medical calificat pentru orice problemă de sănătate.

---

## Contribuții

Dacă doriți să îmbunătățiți algoritmul sau să adăugați noi funcționalități, pull request-urile sunt binevenite!

