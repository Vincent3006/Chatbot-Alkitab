# Nama file: custom_embedder.py

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List
from langchain_core.embeddings import Embeddings

# Ini adalah fungsi helper dari kode1, kita letakkan di sini agar bisa diakses.
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Ini adalah kelas utama kita yang mengimplementasikan "kontrak" dari LangChain
class QwenEmbedder(Embeddings):
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', max_length: int = 8192):
        """
        Inisialisasi model dan tokenizer dari Hugging Face.
        Ini hanya dijalankan sekali saat objek dibuat.
        """
        super().__init__()
        print("Menginisialisasi Custom Qwen Embedder secara manual...")
        self.model_name = model_name
        self.max_length = max_length
        
        # Tentukan perangkat (GPU jika tersedia, jika tidak CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   > Menggunakan perangkat: {self.device}")

        # Muat tokenizer dan model dari Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Pindahkan model ke perangkat yang ditentukan dan set ke mode evaluasi
        self.model.to(self.device)
        self.model.eval()
        print("   > Model dan tokenizer berhasil dimuat.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Metode ini dipanggil oleh ChromaDB saat membuat database.
        Menerima daftar teks dan mengembalikan daftar vektor embedding.
        """
        # Gabungkan instruksi jika diperlukan (opsional, tapi disarankan untuk Qwen)
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        input_texts = [f'Instruct: {task}\nQuery: {text}' for text in texts]

        with torch.no_grad():
            # Tokenisasi batch teks
            batch_dict = self.tokenizer(
                input_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            
            # Pindahkan data tensor ke perangkat
            batch_dict.to(self.device)
            
            # Jalankan model
            outputs = self.model(**batch_dict)
            
            # Dapatkan embedding menggunakan fungsi pooling
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalisasi embedding
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Kembalikan sebagai list of lists di CPU
            return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Metode ini dipanggil saat melakukan retrieval untuk satu pertanyaan.
        Menerima satu teks dan mengembalikan satu vektor embedding.
        """
        # Cara termudah adalah dengan menggunakan kembali embed_documents
        # dengan input berupa list yang berisi satu item.
        single_embedding = self.embed_documents([text])
        return single_embedding[0]