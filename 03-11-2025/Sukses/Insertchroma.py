import os
import re
import numpy as np
import chromadb
import nltk
import pandas as pd 
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import warnings
from chromadb.utils import embedding_functions
warnings.filterwarnings("ignore", category=UserWarning, module='PyPDF2')
KNOWLEDGE_BASE_DIR = "knowledge-base1"
DB_NAME = "tetsing4" # Menggunakan DB baru untuk memastikan tidak ada konflik
HEADER_CROP_PERCENTAGE = 0.15 
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}
EXCEL_OUTPUT_FILENAME = "tetsing4.xlsx"
def preprocess_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def extract_text_with_metadata(pdf_path: str) -> list:
    data = []
    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            print(f"  [PERINGATAN] PDF '{os.path.basename(pdf_path)}' kosong atau tidak dapat dibaca.")
            return []
        raw_text = ""
        print(f"  Mengekstrak teks dari {os.path.basename(pdf_path)} (semua halaman)...")
        for page in reader.pages:
            original_height = float(page.mediabox.height)
            new_top = original_height * (1 - HEADER_CROP_PERCENTAGE)
            page.cropbox.upper_y = new_top
            raw_text += page.extract_text()
    except Exception as e:
        print(f"  [ERROR] Gagal membaca file {os.path.basename(pdf_path)}: {e}")
        return []
    try:
        sentences = re.split(r'(\w+\s\d+:\d+:\s)', raw_text)
        for i in range(1, len(sentences), 2):
            source = sentences[i].strip()
            text = sentences[i+1].strip()
            if text:
                cleaned_text = preprocess_text(text)
                data.append({"text": cleaned_text, "source": source})
    except Exception as e:
        print(f"  [ERROR] Gagal saat memecah teks dari {os.path.basename(pdf_path)}: {e}")
    return data
def process_data_normal_semantic_chunking(data, hf_embeddings, similarity_threshold=0.4, max_characters=1000, batch_size=32):
    if not data:
        return []
    texts = [d['text'] for d in data]
    sources = [d['source'] for d in data]
    print(f"  Membuat embedding untuk {len(texts)} kalimat dalam batch berukuran {batch_size}...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding Batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        if not batch_texts: continue
        batch_embeddings = hf_embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
    if not all_embeddings:
        print("  [PERINGATAN] Tidak ada embedding yang berhasil dibuat.")
        return []
    embeddings = np.array(all_embeddings)
    if embeddings.shape[0] < 2:
        if texts:
            metadata = {
                "source_range": f"{sources[0]}-{sources[-1]}",
                "final_char_count": len(texts[0]),
                "internal_similarity_scores": [],
                "break_similarity_score": -1.0 
            }
            return [{"document": " ".join(texts), "metadata": metadata}]
        return []
    dot_products = np.einsum('ij,ij->i', embeddings[:-1], embeddings[1:])
    norm_products = np.linalg.norm(embeddings[:-1], axis=1) * np.linalg.norm(embeddings[1:], axis=1)
    valid_mask = norm_products > 0
    similarities = np.zeros(len(dot_products))
    similarities[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
    chunks = []
    current_chunk_texts = [texts[0]]
    current_chunk_sources = [sources[0]]
    current_chunk_length = len(texts[0])
    current_chunk_similarities = []
    for i, similarity in enumerate(similarities):
        next_text = texts[i+1]
        next_source = sources[i+1]
        potential_length = current_chunk_length + 1 + len(next_text)
        is_similar = similarity >= similarity_threshold
        within_limit = potential_length <= max_characters
        if is_similar and within_limit:
            current_chunk_texts.append(next_text)
            current_chunk_sources.append(next_source)
            current_chunk_length = potential_length
            current_chunk_similarities.append(float(similarity))
        else:
            metadata = {
                "source_range": f"{current_chunk_sources[0]}-{current_chunk_sources[-1]}",
                "final_char_count": current_chunk_length,
                "internal_similarity_scores": current_chunk_similarities,
                "break_similarity_score": float(similarity)
            }
            chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
            current_chunk_texts = [next_text]
            current_chunk_sources = [next_source]
            current_chunk_length = len(next_text)
            current_chunk_similarities = []
    if current_chunk_texts:
        metadata = {
            "source_range": f"{current_chunk_sources[0]}-{current_chunk_sources[-1]}",
            "final_char_count": current_chunk_length,
            "internal_similarity_scores": current_chunk_similarities,
            "break_similarity_score": -1.0
        }
        chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
    return chunks
if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Direktori '{KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
        exit()
    try: nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: nltk.download('punkt')
    print("Menginisialisasi model embedding (Qwen)...")
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=MODEL_KWARGS, encode_kwargs=ENCODE_KWARGS)
    print("Menginisialisasi fungsi embedding untuk ChromaDB...")
    chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME, device=MODEL_KWARGS.get('device', 'cpu'))
    print(f"Menghubungkan atau membuat database di: '{DB_NAME}'...")
    client = chromadb.PersistentClient(path=DB_NAME)
    collection = client.get_or_create_collection(name="semantic_chunks", embedding_function=chroma_embedding_function)
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"  [PERINGATAN] Tidak ada file PDF yang ditemukan di folder '{KNOWLEDGE_BASE_DIR}'.")
        exit()
    print("\nMemulai proses penambahan data ke database...")
    for pdf_name in tqdm(pdf_files, desc="Memproses semua PDF", unit="file"):
        existing_docs = collection.get(where={"source_file": pdf_name}, limit=1)
        if existing_docs['ids']:
            print(f"\n  [INFO] Melewati {pdf_name}, data sudah ada di database.")
            continue
        print(f"\nMemproses file baru: {pdf_name}")
        pdf_path = os.path.join(KNOWLEDGE_BASE_DIR, pdf_name)
        extracted_data = extract_text_with_metadata(pdf_path)
        if not extracted_data:
            print(f"  Tidak ada data yang diekstrak dari {pdf_name}.")
            continue
        final_chunks = process_data_normal_semantic_chunking(extracted_data, hf_embeddings) 
        if final_chunks:
            ids = [f"{pdf_name}_{i}" for i in range(len(final_chunks))]
            documents = [c['document'] for c in final_chunks]
            print(f"  Membuat embedding final untuk {len(documents)} chunks...")
            embeddings_for_db = hf_embeddings.embed_documents(documents)
            metadatas = []
            for c in final_chunks:
                meta = c['metadata']
                meta['source_file'] = pdf_name
                internal_scores_list = meta.get('internal_similarity_scores', [])
                meta['internal_similarity_scores'] = ", ".join([f"{score:.4f}" for score in internal_scores_list])
                metadatas.append(meta)  
            try:
                collection.add(
                    documents=documents, 
                    embeddings=embeddings_for_db,
                    metadatas=metadatas, 
                    ids=ids
                )
                print(f"  Berhasil! {len(final_chunks)} potongan (chunks) dari {pdf_name} ditambahkan.")
            except Exception as e:
                print(f"  [ERROR] Gagal menambahkan data dari {pdf_name} ke ChromaDB: {e}")
    print("\n" + "="*50)
    print("PROSES PENAMBAHAN DATA SELESAI")
    print("="*50)
    print(f"\nMemulai ekspor keseluruhan database ke file Excel: {EXCEL_OUTPUT_FILENAME}")
    try:
        total_docs_in_db = collection.count()
        if total_docs_in_db == 0:
            print("Database kosong. Tidak ada data untuk diekspor.")
        else:
            print(f"Mengambil {total_docs_in_db} dokumen dari database...")
            all_data = collection.get(limit=total_docs_in_db, include=["documents", "metadatas"])
            data_for_excel = []
            for i in range(len(all_data['ids'])):
                metadata = all_data['metadatas'][i]
                data_for_excel.append({
                    "Nomor": i + 1,
                    "source_file": metadata.get('source_file', 'N/A'),
                    "source_range": metadata.get('source_range', 'N/A'),
                    "isi_ayat": all_data['documents'][i],
                    "jumlah_karakter_chunk": metadata.get('final_char_count', 0),
                    "skor_kemiripan_internal": metadata.get('internal_similarity_scores', 'N/A'), # Tinggal ambil karena sudah jadi string
                    "skor_kemiripan_pemisah": metadata.get('break_similarity_score', 'N/A')
                })
            df = pd.DataFrame(data_for_excel)
            if os.path.exists(EXCEL_OUTPUT_FILENAME):
                os.remove(EXCEL_OUTPUT_FILENAME)
                print(f"File '{EXCEL_OUTPUT_FILENAME}' yang sudah ada telah dihapus.")
            df.to_excel(EXCEL_OUTPUT_FILENAME, index=False)
            print(f"\nBerhasil! Seluruh data dari database telah diekspor ke '{EXCEL_OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan saat mengekspor data ke Excel: {e}")
    print("\n--- Program Selesai ---")