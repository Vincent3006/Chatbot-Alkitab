# utils.py
import os
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict

def preprocess_text(text: str) -> str:
    """Membersihkan teks dengan menghapus karakter baru dan spasi berlebihan"""
    # Ganti multiple newline dengan single space
    text = re.sub(r'\n+', ' ', text)
    # Ganti multiple whitespace dengan single space
    text = re.sub(r'\s+', ' ', text)
    # Hilangkan spasi di awal dan akhir
    return text.strip()

def extract_text_with_metadata(pdf_path: str, header_crop_percentage: float) -> List[Dict]:
    """Mengekstrak teks dari file PDF, memotong header, dan chunking berdasarkan ayat."""
    data = []
    
    try:
        # Baca file PDF
        reader = PdfReader(pdf_path)
        
        # Cek jika PDF kosong
        if not reader.pages:
            print(f"PDF '{os.path.basename(pdf_path)}' kosong atau tidak dapat dibaca.")
            return []
        
        raw_text = ""
        print(f"Mengekstrak teks dari {os.path.basename(pdf_path)}...")
        
        # Proses setiap halaman PDF
        for page in reader.pages:
            # Hitung tinggi baru untuk memotong header
            original_height = float(page.mediabox.height)
            new_top = original_height * (1 - header_crop_percentage)
            page.cropbox.upper_y = new_top
            
            # Ekstrak teks dari halaman yang sudah dipotong
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted
                
    except Exception as e:
        print(f"  [ERROR] Gagal membaca file {os.path.basename(pdf_path)}: {e}")
        return []
    
    try:
        # Split teks berdasarkan pola ayat (contoh: "Yohanes 1:1: ")
        sentences = re.split(r'(\w+\s\d+:\d+:\s)', raw_text)
        
        # Jika tidak ada ayat yang ditemukan, return kosong
        if len(sentences) <= 1:
            return [] 
        
        # Proses setiap pasangan source dan teks
        for i in range(1, len(sentences), 2):
            source = sentences[i].strip()  # Metadata sumber (ayat)
            text = sentences[i+1].strip()  # Isi teks
             
            if text:
                # Bersihkan teks dan tambahkan ke data
                cleaned_text = preprocess_text(text)
                data.append({"text": cleaned_text, "source": source})
                
    except Exception as e:
        print(f"  [ERROR] Gagal saat memecah teks dari {os.path.basename(pdf_path)}: {e}")
    
    return data

def semantic_chunking(data: List[Dict], hf_embeddings, similarity_threshold: float, max_characters: int, batch_size: int) -> List[Dict]:
    """Fungsi semantic chunking yang menyimpan metadata LENGKAP dan dalam format yang BENAR (string)."""
    
    # Cek jika data kosong
    if not data:
        return []
    
    # Pisahkan teks dan metadata sumber
    texts = [d['text'] for d in data]
    sources = [d['source'] for d in data]
    
    print(f"  Membuat embedding untuk {len(texts)} kalimat dalam batch berukuran {batch_size}...")
    all_embeddings = []
    
    # Buat embedding dalam batch untuk efisiensi
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding Batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        if not batch_texts: continue
        
        # Generate embeddings untuk batch saat ini
        batch_embeddings = hf_embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    # Cek jika embedding berhasil dibuat
    if not all_embeddings:
        print("  [PERINGATAN] Tidak ada embedding yang berhasil dibuat.")
        return []
    
    # Konversi ke numpy array untuk perhitungan vektor
    embeddings = np.array(all_embeddings)
    
    # Handle kasus dengan hanya 1 kalimat
    if embeddings.shape[0] < 2:
        if texts:
            metadata = {
                "source_range": sources[0],  # Sumber tunggal
                "final_char_count": len(texts[0]),  # Panjang karakter
                "internal_similarity_scores": "",  # Kosong karena hanya 1 kalimat
                "break_similarity_score": -1.0  # Nilai default
            }
            return [{"document": texts[0], "metadata": metadata}]
        return []
    
    # Hitung similarity antara embedding berurutan
    # Dot product untuk kemiripan kosinus
    dot_products = np.einsum('ij,ij->i', embeddings[:-1], embeddings[1:])
    # Norm product untuk normalisasi
    norm_products = np.linalg.norm(embeddings[:-1], axis=1) * np.linalg.norm(embeddings[1:], axis=1)
    
    # Hindari division by zero
    valid_mask = norm_products > 0
    similarities = np.zeros(len(dot_products))
    similarities[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
    
    # Inisialisasi variabel untuk chunking
    chunks = []
    current_chunk_texts = [texts[0]]
    current_chunk_sources = [sources[0]]
    current_chunk_length = len(texts[0])
    current_chunk_similarities = []
    
    # Proses semantic chunking
    for i, similarity in enumerate(similarities):
        next_text = texts[i+1]
        next_source = sources[i+1]
        potential_length = current_chunk_length + 1 + len(next_text)
        
        # Cek apakah bisa digabung ke chunk saat ini
        if similarity >= similarity_threshold and potential_length <= max_characters:
            # Gabungkan ke chunk saat ini
            current_chunk_texts.append(next_text)
            current_chunk_sources.append(next_source)
            current_chunk_length = potential_length
            current_chunk_similarities.append(float(similarity))
        else:
            # Simpan chunk saat ini dan mulai chunk baru
            internal_scores_str = ", ".join([f"{score:.4f}" for score in current_chunk_similarities])
            metadata = {
                "source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}",  # Range sumber
                "final_char_count": current_chunk_length,  # Total karakter
                "internal_similarity_scores": internal_scores_str,  # Similarity scores sebagai string
                "break_similarity_score": float(similarity)  # Similarity yang menyebabkan break
            }
            chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
            
            # Reset untuk chunk baru
            current_chunk_texts = [next_text]
            current_chunk_sources = [next_source]
            current_chunk_length = len(next_text)
            current_chunk_similarities = []
    
    # Handle chunk terakhir
    if current_chunk_texts:
        internal_scores_str = ", ".join([f"{score:.4f}" for score in current_chunk_similarities])
        metadata = {
            "source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}",
            "final_char_count": current_chunk_length,
            "internal_similarity_scores": internal_scores_str,
            "break_similarity_score": -1.0  # Nilai default untuk chunk terakhir
        }
        chunks.append({"document": " ".join(current_chunk_texts), "metadata": metadata})
    
    return chunks