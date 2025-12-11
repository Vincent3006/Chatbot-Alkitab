#kode2\custom_embedder.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
class QwenEmbedder: 
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * input_mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts or not isinstance(texts, list):
            return []
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=1024
        )
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()
    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]
if __name__ == '__main__':
    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
    kalimat = [
        "Kasih itu sabar;",
    ]
    embeddings = embedder.embed_documents(kalimat)
    print("\nContoh Kalimat:", kalimat[0])
    print("Dimensi Embedding:", len(embeddings[0]))
    print("Contoh Embedding (5 elemen pertama):", np.array(embeddings[0][:5]))
    
    # Contoh embed satu query
    query_embedding = embedder.embed_query("Apa itu kasih?")
    print("\nContoh Query:", "Apa itu kasih?")
    print("Dimensi Embedding Query:", len(query_embedding))
    print("Contoh Embedding Query (5 elemen pertama):", np.array(query_embedding[:5]))

# # --- IMPOR PUSTAKA YANG DIBUTUHKAN ---

# # File: custom_embedder.py (atau tambahkan ke utils.py)
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from typing import List
# import numpy as np

# class QwenEmbedder:
#     # Metode inisialisasi untuk menyiapkan model saat objek dibuat.
#     def __init__(self, model_name: str, device: str = 'cpu'):
#         print(f"  > Memuat Tokenizer dan Model '{model_name}'...")
#         # Memuat tokenizer dari Hugging Face untuk mengubah teks menjadi token ID.
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # Memuat model Transformer dari Hugging Face untuk menghasilkan embedding.
#         self.model = AutoModel.from_pretrained(model_name)
#         # Menentukan perangkat komputasi (CPU atau GPU).
#         self.device = device
#         # Memindahkan model ke perangkat yang ditentukan.
#         self.model.to(self.device)
#         # Mengubah model ke mode evaluasi (inferensi), bukan mode pelatihan.
#         self.model.eval()
#         print("  > Model dan Tokenizer berhasil dimuat.")

#     # Fungsi internal untuk menggabungkan embedding token menjadi satu embedding kalimat.
#     def _mean_pooling(self, model_output, attention_mask):
#         # Mengambil embedding dari setiap token dari output model.
#         token_embeddings = model_output.last_hidden_state
#         # Menyesuaikan bentuk attention mask agar sama dengan bentuk token_embeddings.
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         # Mengalikan embedding dengan mask untuk mengabaikan token padding (hasilnya jadi nol).
#         masked_embeddings = token_embeddings * input_mask_expanded
#         # Menjumlahkan semua embedding token yang tidak di-mask.
#         sum_embeddings = torch.sum(masked_embeddings, 1)
#         # Menjumlahkan nilai mask untuk mendapatkan jumlah token asli.
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         # Menghitung rata-rata embedding dengan membagi jumlah embedding dengan jumlah token.
#         return sum_embeddings / sum_mask

#     # Metode untuk membuat embedding dari daftar dokumen (kalimat).
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # Mengembalikan list kosong jika input tidak valid.
#         if not texts or not isinstance(texts, list):
#             return []
#         # Mengubah teks menjadi token, menambahkan padding, dan memotong jika terlalu panjang.
#         encoded_input = self.tokenizer(
#             texts, 
#             padding=True, 
#             truncation=True, 
#             return_tensors='pt', # Mengembalikan hasil dalam bentuk PyTorch Tensor.
#             max_length=1024
#         )
#         # Memindahkan semua data input (token ID, attention mask) ke perangkat (CPU/GPU).
#         encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
#         # Menonaktifkan perhitungan gradien untuk mempercepat proses inferensi.
#         with torch.no_grad():
#             # Menjalankan model dengan input yang sudah ditokenisasi.
#             model_output = self.model(**encoded_input)
#         # Menggabungkan embedding token menjadi embedding kalimat menggunakan mean pooling.
#         sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
#         # Menormalkan panjang vektor embedding menjadi 1 untuk perbandingan kemiripan yang konsisten.
#         normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#         # Mengembalikan embedding ke CPU, mengubahnya ke format list Python.
#         return normalized_embeddings.cpu().numpy().tolist()

#     # Metode untuk membuat embedding dari satu query (teks tunggal).
#     def embed_query(self, query: str) -> List[float]:
#         # Memanggil embed_documents untuk satu query dan mengambil hasil pertamanya.
#         return self.embed_documents([query])[0]

# # Blok ini hanya akan berjalan jika file ini dieksekusi secara langsung.
# if __name__ == '__main__':
#     # Membuat instance dari kelas QwenEmbedder dengan model yang ditentukan.
#     embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
    
#     # Menyiapkan daftar kalimat yang akan di-embed.
#     kalimat = [
#         "Kasih itu sabar;",
#     ]
    
#     # Menghasilkan embedding untuk daftar kalimat di atas.
#     embeddings = embedder.embed_documents(kalimat)
    
#     # Mencetak hasil untuk verifikasi.
#     print("\nContoh Kalimat:", kalimat[0])
#     print("Dimensi Embedding:", len(embeddings[0])) # Menampilkan jumlah dimensi vektor.
#     print("Contoh Embedding (5 elemen pertama):", np.array(embeddings[0][:5])) # Menampilkan 5 angka pertama dari vektor.
    
#     # Contoh penggunaan untuk satu query.
#     query_embedding = embedder.embed_query("Apa itu kasih?")

#     # Mencetak hasil query untuk verifikasi.
#     print("\nContoh Query:", "Apa itu kasih?")
#     print("Dimensi Embedding Query:", len(query_embedding))
#     print("Contoh Embedding Query (5 elemen pertama):", np.array(query_embedding[:5]))