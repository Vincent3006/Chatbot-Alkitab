# utils.py

import os
import re
import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

def preprocess_text(text: str) -> str:
    """Membersihkan teks """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_with_metadata(pdf_path: str) -> List[Dict]:
    """Mengekstrak teks dari file PDF dan chunking berdasarkan ayat."""
    data = []
    try:
        reader = PdfReader(pdf_path)
        raw_text = ""
        # Proses selain halaman pertama dan terahkir
        if len(reader.pages) > 2:
            for page in reader.pages[1:-1]:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted
        else:
            return [] 
        #Proses Chunk per ayat
        sentences = re.split(r'(\w+\s\d+:\d+\s)', raw_text)
        
        if len(sentences) <= 1:
            clean_text = preprocess_text(raw_text)
            if clean_text:
                data.append({"text": clean_text, "source": "Halaman 2 - Akhir"})
            return data

        for i in range(1, len(sentences), 2):
            source = preprocess_text(sentences[i])
            text = preprocess_text(sentences[i+1])
            if text:
                data.append({"text": text, "source": source})
    except Exception as e:
        print(f"  [ERROR] Gagal memproses file {os.path.basename(pdf_path)}: {e}")
    return data

def semantic_chunking(data: List[Dict], hf_embeddings, similarity_threshold: float = 0.9, max_tokens: int = 512) -> List[Dict]:
    """
    Melakukan chunking semantik dengan batasan jumlah token maksimal per chunk.

    Args:
        data (List[Dict]): List berisi dictionary dengan kunci 'text' dan 'source'.
        hf_embeddings: Model embedding yang memiliki metode .embed_documents().
        similarity_threshold (float): Ambang batas kemiripan kosinus untuk menggabungkan chunk.
        max_tokens (int): Jumlah token maksimal yang diizinkan untuk satu chunk.

    Returns:
        List[Dict]: List berisi chunk yang sudah digabungkan secara semantik.
    """
    if not data:
        return []
        
    texts = [d['text'] for d in data]
    sources = [d['source'] for d in data]
    
    if not texts:
        return []

    embeddings = np.array(hf_embeddings.embed_documents(texts))
    
    chunks = []
    
    current_chunk_texts = [texts[0]]
    current_chunk_sources = [sources[0]]
    current_chunk_tokens = len(texts[0].split())

    for i in range(1, len(texts)):
        # Hitung kemiripan antara teks sebelumnya (terakhir di chunk saat ini) dengan teks baru
        similarity = cosine_similarity(embeddings[i-1].reshape(1, -1), embeddings[i].reshape(1, -1))[0][0]
        
        # Hitung jumlah token dari teks berikutnya yang akan ditambahkan
        next_text_tokens = len(texts[i].split())
        
        if similarity >= similarity_threshold and (current_chunk_tokens + next_text_tokens) <= max_tokens:
            current_chunk_texts.append(texts[i])
            current_chunk_sources.append(sources[i])
            current_chunk_tokens += next_text_tokens
        else:
 
            
            # 1. Simpan chunk yang sudah selesai
            chunks.append({
                "document": " ".join(current_chunk_texts),
                "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
            })
            
            # 2. Mulai chunk baru dengan teks saat ini
            current_chunk_texts = [texts[i]]
            current_chunk_sources = [sources[i]]
            current_chunk_tokens = next_text_tokens # Reset penghitung token
            
    #  menyimpan chunk terakhir yang sedang berjalan
    if current_chunk_texts:
        chunks.append({
            "document": " ".join(current_chunk_texts),
            "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
        })
        
    return chunks