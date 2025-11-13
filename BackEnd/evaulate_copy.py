import pandas as pd
from tqdm import tqdm
from rag_pipeline import CustomRAGPipeline
import numpy as np
import os

# --- KONFIGURASI EVALUASI ---
EVAL_DATASET_PATH = "evaulate.xlsx"
EVAL_OUTPUT_PATH = "evaluation_results.xlsx" 
RETRIEVAL_K = 4 
def load_evaluation_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset evaluasi dari file EXCEL."""
    try:
        df = pd.read_excel(filepath)
        df = df.dropna(subset=['Expected_Chunks_ID'])
        df['Expected_Chunks_ID'] = df['Expected_Chunks_ID'].astype(str) # Pastikan tipe datanya string
        df['Expected_Chunks_ID_List'] = df['Expected_Chunks_ID'].apply(lambda x: x.split('|'))
        print(f"Dataset evaluasi '{filepath}' berhasil dimuat dengan {len(df)} queries.")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File dataset evaluasi tidak ditemukan di '{filepath}'.")
        return None
def calculate_metrics(retrieved_chunks: list, expected_chunks: list):
    """
    Menghitung metrik Precision@k, Recall@k, dan Reciprocal Rank untuk satu query.
    """
    expected_set = set(expected_chunks)
    retrieved_set = set(retrieved_chunks)
    true_positives = len(expected_set.intersection(retrieved_set))
    precision = true_positives / len(retrieved_chunks) if retrieved_chunks else 0
    recall = true_positives / len(expected_chunks) if expected_chunks else 0
    reciprocal_rank = 0.0
    for i, chunk in enumerate(retrieved_chunks):
        if chunk in expected_set:
            reciprocal_rank = 1 / (i + 1)
            break
    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank
    }
def main():
    """Fungsi utama untuk menjalankan proses evaluasi."""
    print("Memulai proses evaluasi pipeline RAG...")
    try:
        rag_pipeline = CustomRAGPipeline()
    except Exception as e:
        print(f"Gagal menginisialisasi RAG Pipeline: {e}")
        return
    eval_df = load_evaluation_data(EVAL_DATASET_PATH)
    if eval_df is None:
        return
    detailed_results = []
    print(f"\nMengevaluasi {len(eval_df)} queries dengan k={RETRIEVAL_K}...")
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Mengevaluasi"):
        query_id = row['Query_ID']
        query_text = row['Query_Text']
        expected_chunks = row['Expected_Chunks_ID_List']
        expected_chunks_str = row['Expected_Chunks_ID']
        retrieved_docs_raw = rag_pipeline.retrieve_documents(query_text, k=RETRIEVAL_K)
        retrieved_source_ranges = [doc['metadata'].get('source_range', 'N/A') for doc in retrieved_docs_raw]
        metrics = calculate_metrics(retrieved_source_ranges, expected_chunks)
        detailed_results.append({
            "Query_ID": query_id,
            "Query_Text": query_text,
            "Expected_Chunks_ID": expected_chunks_str,
            "Retrieved_Chunks_ID": '|'.join(retrieved_source_ranges), 
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "Reciprocal_Rank": metrics['reciprocal_rank']
        })
    if not detailed_results:
        print("Tidak ada hasil untuk diproses.")
        return
    results_df = pd.DataFrame(detailed_results)
    avg_precision = results_df['Precision'].mean()
    avg_recall = results_df['Recall'].mean()
    mrr = results_df['Reciprocal_Rank'].mean()
    summary_data = {
        'Metric': [
            'Average Precision', 
            'Average Recall', 
            'MRR (Mean Reciprocal Rank)',
            'Total Queries Evaluated',
            'Retrieval @k'
        ],
        'Value': [
            f"{avg_precision:.4f}",
            f"{avg_recall:.4f}",
            f"{mrr:.4f}",
            len(results_df),
            RETRIEVAL_K
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    try:
        with pd.ExcelWriter(EVAL_OUTPUT_PATH, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"\n✅ Hasil evaluasi lengkap telah disimpan di '{os.path.abspath(EVAL_OUTPUT_PATH)}'")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan file Excel: {e}")

if __name__ == "__main__":
    main()