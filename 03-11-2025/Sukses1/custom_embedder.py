# File: custom_embedder.py (atau tambahkan ke utils.py)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
class QwenEmbedder:
    def __init__(self, model_name: str, device: str = 'cpu'):
        print(f"  > Memuat Tokenizer dan Model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print("  > Model dan Tokenizer berhasil dimuat.")
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

# --- IMPOR PUSTAKA YANG DIBUTUHKAN ---

# Mengimpor pustaka PyTorch, yang merupakan kerangka kerja utama untuk deep learning.
import torch
# Mengimpor modul fungsional dari PyTorch, seringkali disingkat sebagai F. 
# Ini berisi fungsi-fungsi seperti normalisasi, fungsi aktivasi, dll.
import torch.nn.functional as F
# Mengimpor kelas AutoTokenizer dan AutoModel dari pustaka Transformers (oleh Hugging Face).
# Kelas-kelas ini memungkinkan kita memuat tokenizer dan model yang sesuai secara otomatis berdasarkan nama model.
from transformers import AutoTokenizer, AutoModel
# Mengimpor tipe 'List' dari modul 'typing' untuk memberikan petunjuk tipe (type hinting) pada fungsi.
from typing import List
# Mengimpor pustaka NumPy, yang digunakan untuk operasi numerik. 
# Meskipun di akhir diubah ke list, konversi ke .numpy() adalah langkah perantara yang umum.
import numpy as np

# --- DEFINISI KELAS QWENEMBEDDER ---

# Mendefinisikan sebuah kelas bernama QwenEmbedder.
# Kelas ini akan membungkus semua logika untuk memuat model dan membuat embedding.
class QwenEmbedder:
    # --- METODE INISIALISASI (__init__) ---
    # Ini adalah konstruktor kelas. Metode ini akan dijalankan secara otomatis saat objek QwenEmbedder baru dibuat.
    # 'model_name' adalah nama model dari Hugging Face yang akan digunakan (misalnya, 'Qwen/Qwen-1_8B-Chat').
    # 'device' adalah perangkat tempat model akan dijalankan ('cpu' atau 'cuda' untuk GPU). Default-nya adalah 'cpu'.
    def __init__(self, model_name: str, device: str = 'cpu'):
        # Mencetak pesan ke konsol untuk memberitahu pengguna bahwa proses pemuatan sedang dimulai.
        print(f"  > Memuat Tokenizer dan Model '{model_name}'...")
        # Memuat tokenizer yang telah dilatih sebelumnya dari Hugging Face berdasarkan 'model_name'.
        # Tokenizer bertanggung jawab untuk mengubah teks menjadi angka (token ID) yang dapat dipahami model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Memuat model yang telah dilatih sebelumnya dari Hugging Face.
        # Model ini akan menghasilkan embedding (representasi vektor) dari teks.
        self.model = AutoModel.from_pretrained(model_name)
        # Menyimpan perangkat (CPU/GPU) yang akan digunakan ke dalam variabel instance.
        self.device = device
        # Memindahkan seluruh model ke perangkat yang ditentukan (CPU atau GPU).
        self.model.to(self.device)
        # Mengubah model ke mode evaluasi. Ini penting karena menonaktifkan beberapa lapisan
        # seperti 'dropout' yang hanya digunakan selama pelatihan, sehingga hasilnya konsisten.
        self.model.eval()
        # Mencetak pesan konfirmasi bahwa model dan tokenizer telah berhasil dimuat.
        print("  > Model dan Tokenizer berhasil dimuat.")

    # --- METODE INTERNAL UNTUK MEAN POOLING ---
    # Metode pribadi (diawali dengan '_') untuk melakukan 'mean pooling'.
    # Pooling ini merangkum semua embedding token dari satu kalimat menjadi satu embedding tunggal (vektor).
    def _mean_pooling(self, model_output, attention_mask):
        # Mengambil embedding dari lapisan tersembunyi terakhir (last hidden state) dari output model.
        # Ini adalah matriks di mana setiap baris mewakili embedding untuk satu token dalam teks input.
        token_embeddings = model_output.last_hidden_state
        # Memperluas dimensi 'attention_mask' agar ukurannya sama dengan 'token_embeddings'.
        # 'attention_mask' digunakan untuk mengidentifikasi token mana yang merupakan token asli dan mana yang 'padding'.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Mengalikan embedding token dengan mask yang diperluas. Ini akan membuat embedding dari token padding menjadi nol.
        masked_embeddings = token_embeddings * input_mask_expanded
        # Menjumlahkan semua embedding token (yang tidak di-mask) pada dimensi ke-1 (dimensi token).
        sum_embeddings = torch.sum(masked_embeddings, 1)
        # Menjumlahkan nilai di mask untuk mendapatkan panjang sebenarnya dari setiap teks (tanpa padding).
        # 'torch.clamp' digunakan untuk memastikan nilainya tidak nol untuk menghindari pembagian dengan nol.
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Mengembalikan hasil bagi dari jumlah embedding dengan jumlah mask, yang merupakan rata-rata embedding.
        return sum_embeddings / sum_mask

    # --- METODE UNTUK MEMBUAT EMBEDDING DOKUMEN ---
    # Metode ini mengambil daftar teks (dokumen) dan mengubahnya menjadi daftar embedding.
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Pengecekan sederhana: jika input kosong atau bukan list, kembalikan list kosong.
        if not texts or not isinstance(texts, list):
            return []
        
        # Menggunakan tokenizer untuk memproses daftar teks.
        encoded_input = self.tokenizer(
            texts, 
            padding=True,       # Menambahkan padding ke teks yang lebih pendek agar semua panjangnya sama dalam satu batch.
            truncation=True,    # Memotong teks yang lebih panjang dari 'max_length'.
            return_tensors='pt',# Mengembalikan hasil dalam bentuk tensor PyTorch.
            max_length=1024     # Menetapkan panjang maksimum token menjadi 1024.
        )
        # Memindahkan semua data input yang telah di-encode (token ID, attention mask, dll.) ke perangkat yang sama dengan model.
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        
        # 'with torch.no_grad()' menonaktifkan perhitungan gradien.
        # Ini penting untuk inferensi (bukan pelatihan) karena menghemat memori dan mempercepat proses.
        with torch.no_grad():
            # Menjalankan model dengan input yang sudah di-encode.
            # Tanda '**' membongkar dictionary 'encoded_input' menjadi argumen untuk model.
            model_output = self.model(**encoded_input)
            
        # Memanggil fungsi _mean_pooling untuk mendapatkan satu vektor embedding untuk setiap teks dalam batch.
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Menormalisasi embedding menggunakan normalisasi L2.
        # Ini membuat semua vektor embedding memiliki panjang (magnitude) 1, yang seringkali berguna untuk perbandingan kemiripan.
        normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Mengembalikan embedding yang sudah dinormalisasi sebagai daftar list float.
        # .cpu() -> Pindahkan tensor ke CPU (jika sebelumnya di GPU).
        # .numpy() -> Ubah tensor menjadi array NumPy.
        # .tolist() -> Ubah array NumPy menjadi list Python standar.
        return normalized_embeddings.cpu().numpy().tolist()

    # --- METODE UNTUK MEMBUAT EMBEDDING QUERY ---
    # Metode ini adalah pintasan untuk membuat embedding dari satu string (query).
    def embed_query(self, query: str) -> List[float]:
        # Metode ini hanya memanggil 'embed_documents' dengan query yang dibungkus dalam sebuah list.
        # Kemudian, ia mengambil hasil pertama [0] karena hanya ada satu item dalam input.
        return self.embed_documents([query])[0]