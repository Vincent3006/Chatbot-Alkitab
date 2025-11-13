# inspect_db.py
import chromadb
import os

# --- KONFIGURASI ---
# PASTIKAN PATH INI SAMA PERSIS DENGAN YANG ANDA COPY
# Ini adalah path ke folder database DI DALAM folder BackEnd
DB_FOLDER_NAME = "testing2" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_FOLDER_NAME)
# --------------------


print(f"--- MENGINSPEKSI DATABASE DI LOKASI: {DB_PATH} ---")

if not os.path.exists(DB_PATH):
    print(f"\n[HASIL] FATAL: Direktori database TIDAK DITEMUKAN di path di atas.")
    print("Pastikan Anda sudah meng-copy folder database ke sini.")
else:
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()

        if not collections:
            print("[HASIL] Database ada, tapi KOSONG. Tidak ada koleksi yang ditemukan.")
        else:
            print(f"Ditemukan {len(collections)} koleksi di dalam database ini:")
            for c in collections:
                print(f"  - Nama Koleksi: {c.name}")
            
            # Mari kita periksa koleksi yang relevan
            COLLECTION_NAME = "semantic_chunks"
            print(f"\n--- Menganalisis Koleksi: '{COLLECTION_NAME}' ---")
            
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                
                # Dapatkan 1 item dari koleksi, TERMASUK embeddingnya
                item = collection.get(limit=1, include=["embeddings"])
                
                if not item['ids']:
                    print(f"  [INFO] Koleksi '{COLLECTION_NAME}' ada, tapi tidak berisi data.")
                else:
                    # Ambil embedding dari item pertama
                    embedding_vector = item['embeddings'][0]
                    actual_dimension = len(embedding_vector)
                    
                    print("\n" + "="*50)
                    print(f"  [BUKTI FINAL] Dimensi embedding yang tersimpan di dalam file database ini adalah: {actual_dimension}")
                    print("="*50)

            except Exception as e_coll:
                 print(f"  [ERROR] Tidak dapat menemukan atau menganalisis koleksi '{COLLECTION_NAME}': {e_coll}")
                
    except Exception as e:
        print(f"\n[ERROR] Gagal terhubung atau membaca database. Mungkin file korup atau salah format. Error: {e}")