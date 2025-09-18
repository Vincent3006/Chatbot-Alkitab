# utils.py

import os
import re
import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

def preprocess_text(text: str) -> str:
    """Memberisihkan Teks"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_with_metadata(pdf_path: str) -> List[Dict]:
    """Mengekstrak teks dalam format metadata"""
    data = []
    try:
        reader = PdfReader(pdf_path)
        raw_text = ""
        # Melwati halaman 1 dan 2
        if len(reader.pages) > 2:
            for page in reader.pages[1:-1]:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted
        else:
            return []

        # Pola regex memisahkan sumber
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

def process_data_normal_semantic_chunking(data: List[Dict], hf_embeddings, similarity_threshold: float = 0.9) -> List[Dict]:
    """Melakukan chunking semantik """
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

    for i in range(1, len(texts)):
        similarity = cosine_similarity(embeddings[i-1].reshape(1, -1), embeddings[i].reshape(1, -1))[0][0]
        
        if similarity >= similarity_threshold:
            current_chunk_texts.append(texts[i])
            current_chunk_sources.append(sources[i])
        else:
            chunks.append({
                "document": " ".join(current_chunk_texts),
                "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
            })
            current_chunk_texts = [texts[i]]
            current_chunk_sources = [sources[i]]
            
    if current_chunk_texts:
        chunks.append({
            "document": " ".join(current_chunk_texts),
            "metadata": {"source_range": f"{current_chunk_sources[0]} - {current_chunk_sources[-1]}"}
        })
        
    return chunks