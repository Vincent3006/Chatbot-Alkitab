# File: custom_embedder.py (atau tambahkan ke utils.py)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np

class QwenEmbedder:
    """
    Kelas kustom untuk membuat embedding teks menggunakan model Qwen dari Hugging Face,
    mengimplementasikan proses secara manual tanpa wrapper LangChain.
    """
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Menginisialisasi tokenizer dan model.
        
        Args:
            model_name (str): Nama model di Hugging Face Hub (cth: 'Qwen/Qwen3-Embedding-0.6B').
            device (str): Perangkat untuk menjalankan model ('cpu' atau 'cuda').
        """
        print(f"  > Memuat Tokenizer dan Model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Pindahkan model ke perangkat yang dipilih (CPU atau GPU)
        self.device = device
        self.model.to(self.device)
        self.model.eval() # Set model ke mode evaluasi (penting untuk inference)
        print("  > Model dan Tokenizer berhasil dimuat.")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Melakukan Mean Pooling.
        Mengambil output dari model dan attention mask untuk menghasilkan satu vektor
        yang merepresentasikan seluruh kalimat.
        """
        # Langkah 1: Dapatkan semua embedding token dari output model
        token_embeddings = model_output.last_hidden_state
        
        # Langkah 2: Buat attention mask yang diperluas agar ukurannya cocok dengan token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Langkah 3: Kalikan embedding dengan mask untuk menolkan embedding dari token padding
        masked_embeddings = token_embeddings * input_mask_expanded
        
        # Langkah 4: Jumlahkan embedding yang relevan dan bagi dengan jumlah token yang relevan
        sum_embeddings = torch.sum(masked_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Membuat embedding untuk sekumpulan dokumen (teks).

        Args:
            texts (List[str]): Daftar string teks yang akan di-embed.

        Returns:
            List[List[float]]: Daftar dari embedding, di mana setiap embedding adalah daftar float.
        """
        if not texts or not isinstance(texts, list):
            return []
            
        # 1. TOKENISASI
        # Mengubah teks menjadi format yang bisa dibaca model (Token ID).
        # padding=True -> Menyamakan panjang semua kalimat dalam batch.
        # truncation=True -> Memotong kalimat yang terlalu panjang.
        # return_tensors='pt' -> Mengembalikan hasil sebagai PyTorch Tensors.
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=1024 # Batas panjang token, bisa disesuaikan
        )
        
        # Pindahkan data token ke perangkat yang sama dengan model
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}

        # 2. INFERENCE MODEL
        # Matikan perhitungan gradien untuk mempercepat proses dan menghemat memori.
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 3. POOLING
        # Mengubah output per-token menjadi satu vektor per-kalimat.
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Pindahkan hasil kembali ke CPU dan ubah menjadi list Python biasa
        return normalized_embeddings.cpu().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Membuat embedding untuk satu teks (pertanyaan).

        Args:
            query (str): String pertanyaan.

        Returns:
            List[float]: Embedding untuk pertanyaan tersebut.
        """
        # Prosesnya sama, hanya saja inputnya adalah list dengan satu elemen.
        # [0] di akhir untuk mengambil embedding tunggal dari hasil list.
        return self.embed_documents([query])[0]


# if __name__ == '__main__':
#     embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
#     kalimat = [
#         "Kasih itu sabar;",
#     ]
#     embeddings = embedder.embed_documents(kalimat)
#     print("\nContoh Kalimat:", kalimat[0])
#     print("Dimensi Embedding:", len(embeddings[0]))
#     print("Contoh Embedding (5 elemen pertama):", np.array(embeddings[0][:5]))
    
#     # Contoh embed satu query
#     query_embedding = embedder.embed_query("Apa itu kasih?")
#     print("\nContoh Query:", "Apa itu kasih?")
#     print("Dimensi Embedding Query:", len(query_embedding))
#     print("Contoh Embedding Query (5 elemen pertama):", np.array(query_embedding[:5]))