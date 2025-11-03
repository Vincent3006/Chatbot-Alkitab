# utils.py
import os
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict
def preprocess_text(text: str) -> str:
    """Membersihkan teks """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def extract_text_with_metadata(pdf_path: str, header_crop_percentage: float) -> List[Dict]:
    """Mengekstrak teks dari file PDF, memotong header, dan chunking berdasarkan ayat."""
    data = []
    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            print(f"  [PERINGATAN] PDF '{os.path.basename(pdf_path)}' kosong atau tidak dapat dibaca.")
            return []
        raw_text = ""
        print(f"  Mengekstrak teks dari {os.path.basename(pdf_path)}...")
        for page in reader.pages:
            original_height = float(page.mediabox.height)
            new_top = original_height * (1 - header_crop_percentage)
            page.cropbox.upper_y = new_top
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted
    except Exception as e:
        print(f"  [ERROR] Gagal membaca file {os.path.basename(pdf_path)}: {e}")
        return []
    try:
        sentences = re.split(r'(\w+\s\d+:\d+:\s)', raw_text)
        if len(sentences) <= 1:
            return [] 
        for i in range(1, len(sentences), 2):
            source = sentences[i].strip()
            text = sentences[i+1].strip()
            if text:
                cleaned_text = preprocess_text(text)
                data.append({"text": cleaned_text, "source": source})
    except Exception as e:
        print(f"  [ERROR] Gagal saat memecah teks dari {os.path.basename(pdf_path)}: {e}")
    return data
# def semantic_chunking(data: List[Dict], hf_embeddings, similarity_threshold: float, max_characters: int, batch_size: int) -> List[Dict]:
#     """Fungsi semantic chunking yang lebih canggih dari skrip kedua (DENGAN PERBAIKAN)."""
#     if not data:
#         return []
#     texts = [d['text'] for d in data]
#     sources = [d['source'] for d in data]
#     print(f"  Membuat embedding untuk {len(texts)} kalimat dalam batch berukuran {batch_size}...")
#     all_embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding Batches", unit="batch"):
#         batch_texts = texts[i:i + batch_size]
#         if not batch_texts: continue
#         batch_embeddings = hf_embeddings.embed_documents(batch_texts)
#         all_embeddings.extend(batch_embeddings)
#     if not all_embeddings:
#         print("  [PERINGATAN] Tidak ada embedding yang berhasil dibuat.")
#         return []
#     embeddings = np.array(all_embeddings)
#     if embeddings.shape[0] < 2:
#         if texts:
#             metadata = { "source_range": sources[0] }
#             return [{"document": texts[0], "metadata": metadata}]
#         return []
#     dot_products = np.einsum('ij,ij->i', embeddings[:-1], embeddings[1:])
#     norm_products = np.linalg.norm(embeddings[:-1], axis=1) * np.linalg.norm(embeddings[1:], axis=1)
#     valid_mask = norm_products > 0
#     similarities = np.zeros(len(dot_products))
#     similarities[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
#     chunks = []
#     current_chunk_texts = [texts[0]]
#     current_chunk_sources = [sources[0]]
#     current_chunk_length = len(texts[0])
#     for i, similarity in enumerate(similarities):
#         next_text = texts[i+1]
#         next_source = sources[i+1]
#         potential_length = current_chunk_length + 1 + len(next_text)
#         if similarity >= similarity_threshold and potential_length <= max_characters:
#             current_chunk_texts.append(next_text)
#             current_chunk_sources.append(next_source) 
#             current_chunk_length = potential_length
#         else:
#             chunks.append({
#                 "document": " ".join(current_chunk_texts),
#                 "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
#             })
#             current_chunk_texts = [next_text]
#             current_chunk_sources = [next_source]
#             current_chunk_length = len(next_text)
#     if current_chunk_texts:
#         chunks.append({
#             "document": " ".join(current_chunk_texts),
#             "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
#         })
#     return chunks
def semantic_chunking(data: List[Dict], hf_embeddings, similarity_threshold: float, max_characters: int, batch_size: int) -> List[Dict]:
    """Fungsi semantic chunking yang menyimpan metadata LENGKAP dan dalam format yang BENAR (string)."""
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
                "source_range": sources[0],
                "final_char_count": len(texts[0]),
                "internal_similarity_scores": "", # Pastikan ini string kosong
                "break_similarity_score": -1.0
            }
            return [{"document": texts[0], "metadata": metadata}]
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

        if similarity >= similarity_threshold and potential_length <= max_characters:
            current_chunk_texts.append(next_text)
            current_chunk_sources.append(next_source)
            current_chunk_length = potential_length
            current_chunk_similarities.append(float(similarity))
        else:
            # Ubah list skor menjadi satu string sebelum membuat metadata
            internal_scores_str = ", ".join([f"{score:.4f}" for score in current_chunk_similarities])
            
            metadata = {
                "source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}",
                "final_char_count": current_chunk_length,
                "internal_similarity_scores": internal_scores_str, # Simpan sebagai string
                "break_similarity_score": float(similarity)
            }
            chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
            
            current_chunk_texts = [next_text]
            current_chunk_sources = [next_source]
            current_chunk_length = len(next_text)
            current_chunk_similarities = []
            
    if current_chunk_texts:
        # Lakukan hal yang sama untuk chunk terakhir
        internal_scores_str = ", ".join([f"{score:.4f}" for score in current_chunk_similarities])
        
        metadata = {
            "source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}",
            "final_char_count": current_chunk_length,
            "internal_similarity_scores": internal_scores_str, # Simpan sebagai string
            "break_similarity_score": -1.0
        }
        chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
        
    return chunks