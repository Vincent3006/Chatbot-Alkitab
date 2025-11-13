import pandas as pd  # Digunakan untuk membaca dan menulis file Excel, serta manipulasi data.
from tqdm import tqdm  # Digunakan untuk membuat progress bar visual saat loop berjalan.
from rag_pipeline import CustomRAGPipeline  # Mengimpor kelas RAG kustom Anda yang akan diuji.
import numpy as np  # Library numerik, sering menjadi dependensi pandas.
import os  # Digunakan untuk berinteraksi dengan sistem operasi, contohnya mendapatkan path file absolut.

# --- KONFIGURASI EVALUASI ---
# Variabel global untuk mengatur parameter evaluasi agar mudah diubah.

# Path menuju file Excel yang berisi dataset evaluasi (pertanyaan dan kunci jawaban).
EVAL_DATASET_PATH = "evaulate.xlsx"

# Path untuk menyimpan file Excel yang akan berisi hasil evaluasi.
EVAL_OUTPUT_PATH = "evaluation_results.xlsx" 

# Menentukan 'k', yaitu berapa banyak dokumen teratas yang akan diambil (retrieved) untuk setiap query.
RETRIEVAL_K = 4 

def load_evaluation_data(filepath: str) -> pd.DataFrame:
    """
    Fungsi ini bertanggung jawab untuk memuat dan memproses dataset evaluasi dari file Excel.
    
    Args:
        filepath (str): Lokasi file Excel yang akan dimuat.

    Returns:
        pd.DataFrame: DataFrame pandas yang sudah bersih dan siap digunakan untuk evaluasi, 
                      atau None jika file tidak ditemukan.
    """
    try:
        # Membaca file Excel ke dalam DataFrame pandas.
        df = pd.read_excel(filepath)
        
        # Membersihkan data: menghapus baris yang tidak memiliki kunci jawaban ('Expected_Chunks_ID') karena tidak bisa dievaluasi.
        df = df.dropna(subset=['Expected_Chunks_ID'])
        
        # Memastikan kolom kunci jawaban bertipe data string untuk konsistensi.
        df['Expected_Chunks_ID'] = df['Expected_Chunks_ID'].astype(str)
        
        # Membuat kolom baru 'Expected_Chunks_ID_List' dengan mengubah string "id1|id2" menjadi list ['id1', 'id2'].
        # Ini sangat penting untuk mempermudah perbandingan nanti.
        df['Expected_Chunks_ID_List'] = df['Expected_Chunks_ID'].apply(lambda x: x.split('|'))
        
        print(f"Dataset evaluasi '{filepath}' berhasil dimuat dengan {len(df)} queries.")
        return df
        
    except FileNotFoundError:
        # Menangani error jika file dataset tidak ditemukan di path yang diberikan.
        print(f"[ERROR] File dataset evaluasi tidak ditemukan di '{filepath}'.")
        return None

def calculate_metrics(retrieved_chunks: list, expected_chunks: list):
    """
    Menghitung metrik performa (Precision@k, Recall@k, Reciprocal Rank) untuk satu query tunggal.

    Args:
        retrieved_chunks (list): List ID chunk yang dikembalikan oleh sistem RAG.
        expected_chunks (list): List ID chunk yang benar (kunci jawaban).

    Returns:
        dict: Sebuah dictionary berisi nilai precision, recall, dan reciprocal_rank.
    """
    # Mengubah list menjadi set untuk operasi irisan (intersection) yang lebih efisien.
    expected_set = set(expected_chunks)
    retrieved_set = set(retrieved_chunks)
    
    # Menghitung True Positives: jumlah chunk yang diambil (retrieved) dan juga ada di kunci jawaban (expected).
    true_positives = len(expected_set.intersection(retrieved_set))
    
    # Menghitung Precision@k: Dari semua yang kita ambil, berapa persen yang benar?
    # Rumus: (Jumlah Benar yang Diambil) / (Total yang Diambil)
    precision = true_positives / len(retrieved_chunks) if retrieved_chunks else 0
    
    # Menghitung Recall@k: Dari semua yang seharusnya ditemukan, berapa persen yang berhasil kita temukan?
    # Rumus: (Jumlah Benar yang Diambil) / (Total Jawaban Benar)
    recall = true_positives / len(expected_chunks) if expected_chunks else 0
    
    # Menghitung Reciprocal Rank: Metrik untuk mengukur seberapa cepat jawaban benar pertama ditemukan.
    reciprocal_rank = 0.0
    # Loop melalui chunk yang diambil sesuai urutannya.
    for i, chunk in enumerate(retrieved_chunks):
        # Jika chunk yang diambil ada di dalam set kunci jawaban.
        if chunk in expected_set:
            # Skornya adalah 1 dibagi peringkatnya (indeks i + 1).
            reciprocal_rank = 1 / (i + 1)
            # Hentikan loop setelah menemukan jawaban benar pertama.
            break
            
    # Mengembalikan semua metrik yang telah dihitung dalam format dictionary.
    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank
    }

def main():
    """Fungsi utama untuk menjalankan seluruh alur proses evaluasi."""
    print("Memulai proses evaluasi pipeline RAG...")
    
    try:
        # 1. Inisialisasi: Membuat instance dari pipeline RAG yang akan diuji.
        rag_pipeline = CustomRAGPipeline()
    except Exception as e:
        print(f"Gagal menginisialisasi RAG Pipeline: {e}")
        return # Keluar dari program jika pipeline gagal dibuat.
    
    # 2. Memuat Data: Memanggil fungsi untuk memuat dataset dari file Excel.
    eval_df = load_evaluation_data(EVAL_DATASET_PATH)
    if eval_df is None:
        return # Keluar dari program jika data gagal dimuat.
        
    # List kosong untuk menyimpan hasil detail dari setiap query.
    detailed_results = []
    
    print(f"\nMengevaluasi {len(eval_df)} queries dengan k={RETRIEVAL_K}...")
    
    # 3. Proses Evaluasi: Melakukan loop untuk setiap baris (query) dalam DataFrame evaluasi.
    # `tqdm` akan menampilkan progress bar di terminal.
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Mengevaluasi"):
        # Mengambil informasi yang dibutuhkan dari setiap baris.
        query_id = row['Query_ID']
        query_text = row['Query_Text']
        expected_chunks = row['Expected_Chunks_ID_List']  # Kunci jawaban dalam format list.
        expected_chunks_str = row['Expected_Chunks_ID']   # Kunci jawaban dalam format string (untuk laporan).
        
        # Menjalankan proses retrieval pada pipeline RAG untuk query saat ini.
        retrieved_docs_raw = rag_pipeline.retrieve_documents(query_text, k=RETRIEVAL_K)
        
        # Mengekstrak ID chunk dari hasil retrieval. ID ada di metadata['source_range'].
        retrieved_source_ranges = [doc['metadata'].get('source_range', 'N/A') for doc in retrieved_docs_raw]
        
        # Menghitung metrik dengan membandingkan hasil retrieval dan kunci jawaban.
        metrics = calculate_metrics(retrieved_source_ranges, expected_chunks)
        
        # Menambahkan hasil detail (termasuk metrik) ke dalam list.
        detailed_results.append({
            "Query_ID": query_id,
            "Query_Text": query_text,
            "Expected_Chunks_ID": expected_chunks_str,
            "Retrieved_Chunks_ID": '|'.join(retrieved_source_ranges), 
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "Reciprocal_Rank": metrics['reciprocal_rank']
        })
        
    # Memeriksa apakah ada hasil untuk diproses.
    if not detailed_results:
        print("Tidak ada hasil untuk diproses.")
        return

    # 4. Agregasi Hasil: Mengubah list hasil detail menjadi DataFrame pandas untuk analisis lebih lanjut.
    results_df = pd.DataFrame(detailed_results)
    
    # Menghitung nilai rata-rata untuk setiap metrik dari semua query yang dievaluasi.
    avg_precision = results_df['Precision'].mean()
    avg_recall = results_df['Recall'].mean()
    mrr = results_df['Reciprocal_Rank'].mean()  # Rata-rata dari Reciprocal Rank disebut Mean Reciprocal Rank (MRR).
    
    # Membuat DataFrame baru untuk ringkasan hasil evaluasi.
    summary_data = {
        'Metric': [
            'Average Precision', 
            'Average Recall', 
            'MRR (Mean Reciprocal Rank)',
            'Total Queries Evaluated',
            'Retrieval @k'
        ],
        'Value': [
            f"{avg_precision:.4f}",  # Format angka menjadi 4 desimal.
            f"{avg_recall:.4f}",
            f"{mrr:.4f}",
            len(results_df),
            RETRIEVAL_K
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # 5. Penyimpanan Hasil: Menyimpan hasil ke file Excel.
    try:
        # Menggunakan ExcelWriter untuk menyimpan dua DataFrame ke dalam sheet yang berbeda di satu file.
        with pd.ExcelWriter(EVAL_OUTPUT_PATH, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"\n✅ Hasil evaluasi lengkap telah disimpan di '{os.path.abspath(EVAL_OUTPUT_PATH)}'")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan file Excel: {e}")

    # 6. Pelaporan di Terminal: Mencetak ringkasan hasil evaluasi ke konsol.
    print("\n" + "="*50)
    print("--- RANGKUMAN HASIL EVALUASI RETRIEVAL ---")
    print("="*50)
    print(f"Average Precision : {avg_precision:.4f}")
    print(f"Average Recall    : {avg_recall:.4f}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("="*50)

# Blok ini memastikan bahwa fungsi main() hanya akan dieksekusi ketika file ini dijalankan sebagai skrip utama.
if __name__ == "__main__":
    main()