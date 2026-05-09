import pickle
import pandas as pd
import numpy as np
import joblib
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import List, Dict, Optional
import warnings
start_time = time.time()
warnings.filterwarnings('ignore')
import gradio as gr
import threading
import webview
class OptimizedMedicalDiagnosisSystem:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        print("Incarca modelul sentence transformer...")
        alternative_models = [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
        self.model = None
        for model in alternative_models:
            try:
                print(f"Incerc sa incarc modelul: {model}")
                self.model = SentenceTransformer(model)
                print(f"Model incarcat cu succes: {model}")
                break
            except Exception as e:
                print(f"Eroare la incarcarea modelului {model}: {str(e)}")
                continue
        if self.model is None:
            print("Nu s-a putut incarca niciun model online. Incerc solutia offline...")
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.use_offline_model = True
                self.tfidf_model = TfidfVectorizer(max_features=300, stop_words='english')
                print("Folosesc model offline TF-IDF ca backup")
            except Exception as e:
                raise RuntimeError(f"Nu s-a putut incarca niciun model: {str(e)}")
        else:
            self.use_offline_model = False
        self.nn_model = None
        self.df = None
        self.embeddings_matrix = None
        self.pca = None
        self.symptom_columns = None
        self.disease_symptoms_cache = {}
    def load_data_from_csv(self, csv_path: str):
        """Incarca datele din CSV cu optimizari"""
        print("Procesez datele medicale din CSV...")
        self.df = pd.read_csv(csv_path, low_memory=False)
        """initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['diseases'])
        print(f"Eliminate {initial_count - len(self.df)} duplicate")"""
        self.symptom_columns = [col for col in self.df.columns if col != 'diseases']
        for col in self.symptom_columns:
            self.df[col] = self.df[col].astype('int8')
        print(f"Incarcate {len(self.df)} boli unice")
        print(f"Numarul de simptome: {len(self.symptom_columns)}")
        self._precompute_symptom_data()
        return self
    def _precompute_symptom_data(self, force_recompute=False,
                                 df_file="cached_symptom_data.pkl",
                                 cache_file="cached_symptom_cache.pkl"):
        """
        Incarca rapid datele pre-calculate din fisierul pickle, daca exista.
        Daca nu, le calculeaza si salveaza in doua fisiere: unul pentru dataframe, altul pentru cache.
        """
        if not force_recompute and os.path.exists(df_file) and os.path.exists(cache_file):
            print(f"📂 Incarc datele pre-calculate din '{df_file}' si '{cache_file}'...")
            self.df = pd.read_pickle(df_file)
            with open(cache_file, "rb") as f:
                self.disease_symptoms_cache = pickle.load(f)
            print("✅ Incarcare completa.")
            return
        print("Pre-calculez datele simptomelor...")
        symptom_names_clean = [col.replace('_', ' ') for col in self.symptom_columns]
        active_symptoms_df = self.df[self.symptom_columns] == 1
        self.df['simptome_text'] = active_symptoms_df.apply(
            lambda row: ', '.join(np.array(symptom_names_clean)[row.values]), axis=1
        )
        self.df['numar_simptome'] = active_symptoms_df.sum(axis=1)
        self.disease_symptoms_cache = {
            row['diseases'].lower(): [
                col.replace('_', ' ') for col in self.symptom_columns if row[col] == 1
            ]
            for _, row in self.df.iterrows()
        }
        print(f"Salvez datele in '{df_file}' si cache-ul in '{cache_file}'...")
        self.df.to_pickle(df_file)
        with open(cache_file, "wb") as f:
            pickle.dump(self.disease_symptoms_cache, f)
        print(f"Pre-calculare completa. {len(self.df)} boli procesate.")
    def save_pca(self, path: str = "pca_model.pkl"):
        if self.pca is None:
            raise ValueError("Nu exista model PCA de salvat.")
        joblib.dump(self.pca, path)
        print(f"PCA salvat la: {path}")
    def load_pca(self, path: str = "pca_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fisierul PCA nu exista: {path}")
        self.pca = joblib.load(path)
        print(f"PCA incarcat din: {path}")
    def save_embeddings(self, path: str = "embeddings.npy"):
        if self.embeddings_matrix is None:
            raise ValueError("Nu exista embeddings de salvat.")
        np.save(path, self.embeddings_matrix)
        print(f"Embeddings salvate la: {path}")
    def load_embeddings(self, path: str = "embeddings.npy"):
        if self.df is None:
            raise ValueError("Datele trebuie incarcate inainte de a incarca embeddings-urile.")
        self.embeddings_matrix = np.load(path)
        print(f"Embeddings incarcate din: {path}")
    def save_index(self, path: str = "nn_model.pkl"):
        if self.nn_model is None:
            raise ValueError("Modelul NearestNeighbors nu a fost antrenat.")
        joblib.dump(self.nn_model, path)
        print(f"Indexul a fost salvat la: {path}")
    def load_index(self, path: str = "nn_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fisierul index nu exista: {path}")
        self.nn_model = joblib.load(path)
        print(f"Indexul a fost incarcat din: {path}")
    def generate_embeddings(self, batch_size: int = 128):
        """Genereaza embeddings cu batch size optimizat"""
        if self.df is None:
            raise ValueError("Trebuie sa incarci datele mai intai!")
        print("\nGenerez embeddings pentru simptome...")
        clean_texts = [text.lower().strip() for text in self.df["simptome_text"]]
        embeddings = self.model.encode(
            clean_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        self.embeddings_matrix = embeddings
        print("Embeddings generate cu succes!")
        return self
    def reduce_dimensions_with_pca(self, n_components: int = 500):
        """Reduce dimensiunea embeddings-urilor cu PCA"""
        if self.embeddings_matrix is None:
            raise ValueError("Trebuie sa generezi embeddings mai intai!")
        original_dims = self.embeddings_matrix.shape[1]
        n_components = min(n_components, original_dims, len(self.df) - 1)
        print(f"\nReduc dimensiunea embeddings-urilor de la {original_dims} la {n_components}...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_matrix = self.pca.fit_transform(self.embeddings_matrix)
        variance_explained = self.pca.explained_variance_ratio_.sum()
        print(f"PCA aplicat! Varianta explicata: {variance_explained:.3f}")
        return self
    def build_index(self, n_neighbors: int = 5):
        """Construieste indexul pentru cautare"""
        if self.embeddings_matrix is None:
            raise ValueError("Trebuie sa generezi embeddings mai intai!")
        actual_neighbors = min(n_neighbors, len(self.df))
        algorithm = 'brute'
        self.nn_model = NearestNeighbors(
            n_neighbors=actual_neighbors,
            metric='cosine',
            algorithm=algorithm,
            n_jobs=-1
        )
        self.nn_model.fit(self.embeddings_matrix)
        print(f"Index construit cu {len(self.df)} inregistrari folosind {algorithm}")
        return self
    def diagnose(self, input_symptoms: str, top_k: int = 3) -> List[Dict]:
        """Diagnosticheaza rapid folosind cache-ul"""
        if self.nn_model is None:
            raise ValueError("Trebuie sa construiesti indexul mai intai!")
        query_text = f"query: {input_symptoms}"
        input_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        if self.pca is not None:
            input_embedding = self.pca.transform(input_embedding)
        distances, indices = self.nn_model.kneighbors(
            input_embedding,
            n_neighbors=min(top_k * 2, len(self.df))
        )
        results = self._process_results_optimized(distances[0], indices[0], top_k)
        return results
    def _process_results_optimized(self, distances, indices, top_k: int) -> List[Dict]:
        """Proceseaza rezultatele mai eficient"""
        unique_diseases = {}
        for dist, idx in zip(distances, indices):
            similarity_score = 1 - dist
            row = self.df.iloc[idx]
            disease = row['diseases']
            if disease not in unique_diseases or similarity_score > unique_diseases[disease]['similarity_score']:
                unique_diseases[disease] = {
                    'boala': disease,
                    'simptome_referinta': row['simptome_text'],
                    'numar_simptome': int(row['numar_simptome']),
                    'similarity_score': round(float(similarity_score), 3),
                    'confidence': self._get_confidence_level(similarity_score)
                }
        sorted_results = sorted(
            unique_diseases.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:top_k]
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1
        return sorted_results
    def _get_confidence_level(self, score: float) -> str:
        """Determina nivelul de incredere"""
        if score >= 0.8:
            return "Foarte mare"
        elif score >= 0.6:
            return "Mare"
        elif score >= 0.4:
            return "Medie"
        else:
            return "Scazuta"
    def print_diagnosis(self, results: List[Dict], input_symptoms: str):
        """Afiseaza diagnosticul"""
        print("\n" + "=" * 80)
        print("SISTEM DE DIAGNOZA MEDICALA OPTIMIZAT")
        print("=" * 80)
        print(f"Simptome introduse: {input_symptoms}")
        print("\nDiagnostice posibile:")
        print("-" * 80)
        for result in results:
            print(f"\n{result['rank']}. {result['boala'].upper()}")
            print(f"Scor similaritate: {result['similarity_score']} ({result['confidence']} incredere)")
            print(f"Numar simptome in DB: {result['numar_simptome']}")
            symptoms_display = result['simptome_referinta']
            if len(symptoms_display) > 150:
                symptoms_display = symptoms_display[:150] + "..."
            print(f"Simptome de referinta: {symptoms_display}")
        print("\n" + "=" * 80)
        print("ATENTIE: Acest sistem este doar pentru informare!")
        print("   Consultati intotdeauna un medic pentru diagnostic real!")
        print("=" * 80)
    def get_disease_symptoms(self, disease_name: str) -> List[str]:
        """Returneaza lista de simptome pentru o boala specifica folosind cache-ul"""
        return self.disease_symptoms_cache.get(disease_name.lower(), [])
    def save_complete_model(self, base_path: str = "medical_model"):
        """Salveaza toate componentele modelului"""
        self.save_embeddings(f"{base_path}_embeddings.npy")
        if self.pca is not None:
            self.save_pca(f"{base_path}_pca.pkl")
        self.save_index(f"{base_path}_index.pkl")
        self.df.to_pickle(f"{base_path}_dataframe.pkl")
        print(f"Model complet salvat cu prefixul: {base_path}")
    def load_complete_model(self, base_path: str = "medical_model"):
        """Incarca toate componentele modelului"""
        self.df = pd.read_pickle(f"{base_path}_dataframe.pkl")
        self.symptom_columns = [col for col in self.df.columns if
                                col not in ('diseases', 'simptome_text', 'numar_simptome')]
        self._precompute_symptom_data()
        self.load_embeddings(f"{base_path}_embeddings.npy")
        pca_path = f"{base_path}_pca.pkl"
        if os.path.exists(pca_path):
            self.load_pca(pca_path)
        self.load_index(f"{base_path}_index.pkl")
        print(f"Model complet incarcat cu prefixul: {base_path}")
def main(fast_mode: bool = True):
    """Functia principala optimizata"""
    system = OptimizedMedicalDiagnosisSystem()
    if os.path.exists("medical_model_dataframe.pkl"):
        print("Incarca modelul pre-antrenat...")
        system.load_complete_model("medical_model")
    else:
        print("Antreneaza modelul pentru prima data...")
        system.load_data_from_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        system.generate_embeddings(batch_size=64)
        system.reduce_dimensions_with_pca(n_components=50)
        system.build_index(n_neighbors=5)
        system.save_complete_model("medical_model")
    test_symptoms = [
        "am febra si ma doare capul de doua zile",
        "anxiety nervousness chest pain palpitations",
        "stomach pain nausea vomiting",
        "cough fever shortness of breath"
    ]
    print("\nTestez diagnosticele...")
    for symptoms in test_symptoms:
        rezultate = system.diagnose(symptoms, top_k=3)
        system.print_diagnosis(rezultate, symptoms)
        print("\n" + "=" * 50 + "\n")
    end_time = time.time()
    print(f"impul total de executie: {end_time - start_time:.2f} secunde")
    def chat(symptom_input, chat_history=[]):
        rezultate = system.diagnose(symptom_input, top_k=3)
        raspuns = "Diagnostic posibil:\n"
        for item in rezultate:
            raspuns += f"- {item['boala']} (incredere: {item['confidence']}, scor: {item['similarity_score']})\n"
        chat_history.append((symptom_input, raspuns))
        return "", chat_history
    with gr.Blocks() as demo:
        gr.Markdown("##Asistent Diagnostic Medical")
        chatbot = gr.Chatbot()
        with gr.Row():
            msg = gr.Textbox(placeholder="Scrie simptomele tale aici...", scale=4)
            send_btn = gr.Button("Trimite", scale=1)
        clear_btn = gr.Button("Sterge chatul")
        send_btn.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    def run_gradio():
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860, quiet=True)
    threading.Thread(target=run_gradio, daemon=True).start()
    webview.create_window("Asistent Diagnostic Medical", "http://127.0.0.1:7860")
    webview.start()
if __name__ == "__main__":
    main()
