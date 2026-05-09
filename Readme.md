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

## 🛠️ Instalare și Cerințe

Pentru a rula acest proiect, aveți nevoie de Python 3.8+ și următoarele biblioteci:

```bash
pip install pandas numpy scikit-learn sentence-transformers joblib gradio pywebview
