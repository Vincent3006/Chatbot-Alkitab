# rag_pipeline.py

# --- Impor Pustaka yang Dibutuhkan ---
import os  # Untuk berinteraksi dengan sistem operasi, seperti mengecek path file.
import pandas as pd  # Pustaka populer untuk manipulasi dan analisis data, digunakan untuk ekspor ke Excel.
from tqdm import tqdm  # Untuk menampilkan progress bar yang informatif saat memproses file.
from langchain_community.vectorstores import Chroma  # Implementasi database vektor ChromaDB dari LangChain.
from langchain_community.embeddings import HuggingFaceEmbeddings # Meskipun tidak digunakan, ini adalah kelas standar untuk model embedding.
from langchain_core.documents import Document  # Struktur data standar LangChain untuk merepresentasikan sepotong teks.
from langchain_core.prompts import PromptTemplate  # Untuk membuat template prompt yang bisa diisi dengan variabel.
from langchain_google_genai import ChatGoogleGenerativeAI # Meskipun tidak digunakan, ini adalah kelas untuk model chat dari Google.
import config  # Mengimpor file konfigurasi (misal: nama path, template prompt, dll).
from langchain_ollama import ChatOllama  # Untuk berinteraksi dengan model bahasa besar (LLM) yang berjalan lokal via Ollama.
from utils import extract_text_with_metadata, semantic_chunking  # Mengimpor fungsi bantuan kustom dari file utils.py.
from custom_embedder import QwenEmbedder  # Mengimpor kelas pembungkus kustom untuk model embedding Qwen.

class CustomRAGPipeline:
    """
    Kelas utama yang mengorkestrasi seluruh alur kerja RAG (Retrieval-Augmented Generation).
    Mulai dari inisialisasi, pembuatan database, hingga menjawab pertanyaan.
    """
    def __init__(self):
        """
        Konstruktor kelas. Metode ini akan langsung dieksekusi saat objek dibuat.
        """
        print("1. Menginisialisasi Model dan Database...")
        # Mempersiapkan model untuk mengubah teks menjadi vektor (embedding).
        self.embedding_model = self._initialize_embeddings()
        # Memuat database vektor yang sudah ada atau membuat yang baru jika belum ada.
        self.vectorstore = self._load_or_create_vectorstore()
        # Mempersiapkan model bahasa besar (LLM) yang akan menghasilkan jawaban.
        self.llm = self._initialize_llm()
        # Inisialisasi daftar kosong untuk menyimpan riwayat percakapan.
        self.chat_history = []
        # Membuat template prompt utama yang akan diisi dengan konteks, riwayat, dan pertanyaan.
        self.main_prompt = PromptTemplate(
            template=config.PROMPT_TEMPLATE, 
            input_variables=["chat_history", "context", "question"]
        )

    def _initialize_embeddings(self):
        """
        Mempersiapkan dan mengembalikan instance dari model embedding kustom (QwenEmbedder).
        """
        return QwenEmbedder(
            model_name=config.MODEL_NAME, # Nama model diambil dari file konfigurasi.
            device=config.MODEL_KWARGS.get('device', 'cpu') # Menggunakan device (GPU/CPU) dari konfigurasi.
        )

    def _initialize_llm(self):
        """
        Mempersiapkan dan mengembalikan instance dari LLM yang berjalan secara lokal via Ollama.
        """
        return ChatOllama(
            model="qwen2.5:3b",  # Menentukan model spesifik yang akan digunakan.
            temperature=0.3      # Mengatur tingkat "kreativitas" model. Nilai rendah membuatnya lebih faktual.
        )

    def _export_database_to_excel(self, vectorstore):
        """
        Mengekspor seluruh isi database vektor ke dalam sebuah file Excel untuk inspeksi dan debugging.
        """
        print("\n" + "="*50)
        print(f"MEMULAI EKSPOR DATABASE KE: {config.EXCEL_OUTPUT_FILENAME}")
        try:
            # Mengakses koleksi data mentah dari dalam objek vectorstore Chroma.
            collection = vectorstore._collection
            # Menghitung total dokumen yang ada di dalam koleksi.
            total_docs = collection.count()
            if total_docs == 0:
                print("Database kosong. Tidak ada data untuk diekspor.")
                return

            print(f"Mengambil {total_docs} dokumen dari database...")
            # Mengambil semua data (dokumen dan metadata) dari koleksi.
            all_data = collection.get(limit=total_docs, include=["documents", "metadatas"])
            
            data_for_excel = [] # Daftar untuk menampung data yang akan diubah menjadi DataFrame.
            # Looping melalui setiap dokumen yang berhasil diambil.
            for i in range(len(all_data['ids'])):
                metadata = all_data['metadatas'][i]
                # Menambahkan data yang sudah diformat ke dalam daftar.
                data_for_excel.append({
                    "Nomor": i + 1,
                    "source_file": metadata.get('source_pdf', 'N/A'),
                    "source_range": metadata.get('source_range', 'N/A'),
                    "isi_ayat": all_data['documents'][i],
                    "jumlah_karakter_chunk": metadata.get('final_char_count', 0),
                    "skor_kemiripan_internal": metadata.get('internal_similarity_scores', ''), 
                    "skor_kemiripan_pemisah": metadata.get('break_similarity_score', -1.0)
                })
            
            # Membuat DataFrame pandas dari daftar data.
            df = pd.DataFrame(data_for_excel)
            # Jika file Excel sudah ada, hapus terlebih dahulu untuk memastikan file baru yang dibuat.
            if os.path.exists(config.EXCEL_OUTPUT_FILENAME):
                os.remove(config.EXCEL_OUTPUT_FILENAME)
                print(f"File '{config.EXCEL_OUTPUT_FILENAME}' yang sudah ada telah dihapus.")
            
            # Menyimpan DataFrame ke file Excel.
            df.to_excel(config.EXCEL_OUTPUT_FILENAME, index=False)
            print(f"BERHASIL! Seluruh data dari database telah diekspor ke '{config.EXCEL_OUTPUT_FILENAME}'.")
        except Exception as e:
            # Menangani jika terjadi error saat proses ekspor.
            print(f"[ERROR] Gagal mengekspor data ke Excel: {e}")
        finally:
            # Blok ini akan selalu dieksekusi, baik berhasil maupun gagal.
            print("="*50 + "\n")

    def _load_or_create_vectorstore(self):
        """
        Fungsi inti untuk memuat database vektor jika ada, atau membuatnya dari awal jika tidak ada.
        """
        COLLECTION_NAME = "semantic_chunks" # Nama koleksi di dalam database Chroma.
        # Memeriksa apakah direktori database sudah ada di sistem.
        if os.path.exists(config.DB_NAME):
            print(f"2. Database '{config.DB_NAME}' ditemukan. Memuat...")
            # Jika ada, muat database yang sudah ada.
            return Chroma(
                persist_directory=config.DB_NAME,
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME
            )
        else:
            # Jika tidak ada, mulai proses pembuatan database baru.
            print(f"2. Database '{config.DB_NAME}' tidak ditemukan. Membuat baru...")
            if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
                raise RuntimeError(f"Direktori '{config.KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
            
            # Mencari semua file dengan ekstensi .pdf di direktori sumber pengetahuan.
            pdf_files = [f for f in os.listdir(config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
            if not pdf_files:
                raise RuntimeError(f"Tidak ada PDF di '{config.KNOWLEDGE_BASE_DIR}'.")
            
            all_final_chunks = [] # Daftar untuk menampung semua chunk dari semua file PDF.
            # Looping melalui setiap file PDF dengan progress bar (tqdm).
            for pdf_name in tqdm(pdf_files, desc="Memproses PDF untuk database"):
                pdf_path = os.path.join(config.KNOWLEDGE_BASE_DIR, pdf_name)
                # Ekstrak teks mentah beserta metadatanya dari satu file PDF.
                extracted_data = extract_text_with_metadata(pdf_path, config.HEADER_CROP_PERCENTAGE)
                if not extracted_data: continue # Lanjut ke PDF berikutnya jika tidak ada data yang diekstrak.

                # Lakukan pemecahan teks menjadi potongan-potongan bermakna (semantic chunking).
                final_chunks = semantic_chunking(
                    extracted_data, 
                    self.embedding_model, # Model embedding diperlukan untuk menghitung kemiripan antar kalimat.
                    config.SIMILARITY_THRESHOLD,
                    config.MAX_CHARACTERS_PER_CHUNK,
                    config.BATCH_SIZE_EMBEDDING
                )

                if final_chunks:
                    # Menambahkan nama file PDF sebagai sumber ke metadata setiap chunk.
                    for chunk in final_chunks: 
                        chunk['metadata']['source_pdf'] = pdf_name
                    # Menambahkan chunk dari PDF ini ke daftar gabungan.
                    all_final_chunks.extend(final_chunks)

            if not all_final_chunks:
                raise RuntimeError("Tidak ada chunk yang berhasil dibuat dari PDF manapun.")
            
            # Mengubah format chunk menjadi objek 'Document' yang dipahami oleh LangChain.
            langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
            
            print("\n   > Membuat dan menyimpan database vektor...")
            # Membuat database Chroma dari daftar dokumen. Proses ini otomatis menghitung embedding.
            vectorstore = Chroma.from_documents(
                documents=langchain_documents,
                embedding=self.embedding_model,
                persist_directory=config.DB_NAME, 
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"} 
            )
            print(f"   > Database berhasil dibuat dengan {vectorstore._collection.count()} dokumen.")
            
            # Setelah database dibuat, ekspor isinya ke Excel.
            self._export_database_to_excel(vectorstore)
            return vectorstore

    def _format_chat_history(self):
        """
        Mengubah riwayat percakapan (list of tuples) menjadi format string tunggal.
        """
        if not self.chat_history:
            return "Tidak ada riwayat percakapan."
        # Menggabungkan setiap pasangan tanya-jawab menjadi format "Manusia: ... Asisten: ...".
        return "\n".join([f"Manusia: {q}\nAsisten: {a}" for q, a in self.chat_history])

    def invoke(self, question: str):
        """
        Metode utama untuk memproses pertanyaan pengguna dan menghasilkan jawaban.
        """
        standalone_question = question
        print(f"   > Pertanyaan yang akan dicari di database: '{standalone_question}'")
        print("   > Melakukan retrieval dokumen dengan skor similarity murni...")
        # Mencari dokumen yang paling relevan dengan pertanyaan dari database vektor.
        retrieved_results_with_scores = self.retrieve_documents(standalone_question, k=10)
        
        # Mengubah hasil pencarian menjadi objek 'Document' LangChain.
        documents_for_prompt = [
            Document(page_content=result['content'], metadata=result['metadata']) 
            for result in retrieved_results_with_scores
        ]
        
        print("   > Memformat konteks Alkitab untuk LLM...")
        # Template untuk memformat setiap dokumen yang ditemukan.
        document_prompt = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
        context_parts = [] # Daftar untuk menyimpan string dari setiap dokumen yang sudah diformat.
        for doc in documents_for_prompt:
            doc_metadata = {
                "source_range": doc.metadata.get('source_range', 'N/A'),
                "page_content": doc.page_content 
            }
            # Mengisi template dokumen dengan metadata dan konten dari dokumen yang ditemukan.
            formatted_doc = document_prompt.format(**doc_metadata)
            context_parts.append(formatted_doc)
        
        # Menggabungkan semua dokumen yang diformat menjadi satu string konteks besar.
        context = "\n".join(context_parts)
        
        print("   > Memformat riwayat percakapan untuk prompt final...")
        # Memformat riwayat percakapan untuk dimasukkan ke dalam prompt.
        formatted_history = self._format_chat_history()
        
        # Augementd dengan : riwayat, konteks, dan pertanyaan.
        final_prompt_for_llm = self.main_prompt.format(
            chat_history=formatted_history, 
            context=context, 
            question=question
        )
        
        # Mencetak prompt final .
        print("\n" + "="*50)
        print("--- PROMPT FINAL YANG DIKIRIM KE LLM ---")
        print(final_prompt_for_llm)
        print("="*50 + "\n")
        
        print("   > Menghasilkan jawaban akhir...")
        # Membuat 'rantai' (chain) yang menghubungkan prompt dengan LLM.
        rag_chain = self.main_prompt | self.llm
        # Menjalankan chain dengan input yang sudah disiapkan untuk mendapatkan jawaban.
        answer = rag_chain.invoke({
            "chat_history": formatted_history,
            "context": context,
            "question": question
        }).content
        
        # Menyimpan pertanyaan dan jawaban baru ke dalam riwayat percakapan.
        self.chat_history.append((question, answer))
        
        # Mengembalikan jawaban dan dokumen sumber sebagai hasil akhir.
        return {
            "answer": answer,
            "source_documents": retrieved_results_with_scores
        }

    def retrieve_documents(self, question: str, k: int = 4):
        """
        Melakukan pencarian kemiripan di database vektor berdasarkan pertanyaan.
        """
        print(f"   > Melakukan retrieval MURNI untuk pertanyaan: '{question}'")
        # Mengubah pertanyaan teks menjadi representasi vektor (embedding).
        question_embedding = self.embedding_model.embed_query(question) 
        
        # Melakukan query langsung ke koleksi ChromaDB dengan embedding pertanyaan.
        query_results = self.vectorstore._collection.query(
            query_embeddings=[question_embedding],
            n_results=k # Jumlah dokumen teratas yang ingin diambil.
        )

        results_list = [] # Daftar untuk menyimpan hasil yang telah diproses.
        # Mengekstrak komponen-komponen hasil dari respons query ChromaDB.
        ids = query_results.get('ids', [[]])[0]
        documents = query_results.get('documents', [[]])[0]
        metadatas = query_results.get('metadatas', [[]])[0]
        distances = query_results.get('distances', [[]])[0] # Jarak kosinus.
        
        # Looping melalui setiap hasil yang dikembalikan.
        for i in range(len(ids)):
            # Mengonversi jarak (distance) menjadi skor kemiripan (similarity). Semakin kecil jarak, semakin tinggi kemiripan.
            similarity_score = 1 - distances[i]
            # Menambahkan hasil yang sudah rapi ke dalam daftar.
            results_list.append({
                "content": documents[i],
                "metadata": metadatas[i],
                "score": similarity_score 
            })
            
        print(f"   > Ditemukan {len(results_list)} dokumen relevan dengan skor similarity murni.")
        return results_list

def initialize_rag_pipeline():
    """
    Fungsi pembantu (factory function) untuk membuat dan mengembalikan instance dari CustomRAGPipeline.
    """
    return CustomRAGPipeline()