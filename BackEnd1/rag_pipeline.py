# rag_pipeline.py
import os
import pandas as pd 
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import config
from langchain_ollama import ChatOllama
from utils import extract_text_with_metadata, semantic_chunking
from custom_embedder import QwenEmbedder 
class CustomRAGPipeline:
    def __init__(self):
        print("1. Menginisialisasi Model dan Database...")
        self.embedding_model = self._initialize_embeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.llm = self._initialize_llm()
        self.chat_history = []
        self.main_prompt = PromptTemplate(
            template=config.PROMPT_TEMPLATE, 
            input_variables=["chat_history", "context", "question"]
        )
    def _initialize_embeddings(self):
        return QwenEmbedder(
            model_name=config.MODEL_NAME,
            device=config.MODEL_KWARGS.get('device', 'cpu') 
        )
    def _initialize_llm(self):
        return ChatOllama(
            model="qwen2.5:3b",
            temperature=0.3
        )
    def _export_database_to_excel(self, vectorstore):
        """Mengekspor seluruh isi database ke file Excel dengan kolom yang detail."""
        print("\n" + "="*50)
        print(f"MEMULAI EKSPOR DATABASE KE: {config.EXCEL_OUTPUT_FILENAME}")
        try:
            collection = vectorstore._collection
            total_docs = collection.count()
            if total_docs == 0:
                print("Database kosong. Tidak ada data untuk diekspor.")
                return
            print(f"Mengambil {total_docs} dokumen dari database...")
            all_data = collection.get(limit=total_docs, include=["documents", "metadatas"])
            data_for_excel = []
            for i in range(len(all_data['ids'])):
                metadata = all_data['metadatas'][i]
                data_for_excel.append({
                    "Nomor": i + 1,
                    "source_file": metadata.get('source_pdf', 'N/A'),
                    "source_range": metadata.get('source_range', 'N/A'),
                    "isi_ayat": all_data['documents'][i],
                    "jumlah_karakter_chunk": metadata.get('final_char_count', 0),
                    "skor_kemiripan_internal": metadata.get('internal_similarity_scores', ''), 
                    "skor_kemiripan_pemisah": metadata.get('break_similarity_score', -1.0)
                })
            df = pd.DataFrame(data_for_excel)
            if os.path.exists(config.EXCEL_OUTPUT_FILENAME):
                os.remove(config.EXCEL_OUTPUT_FILENAME)
                print(f"File '{config.EXCEL_OUTPUT_FILENAME}' yang sudah ada telah dihapus.")
            df.to_excel(config.EXCEL_OUTPUT_FILENAME, index=False)
            print(f"BERHASIL! Seluruh data dari database telah diekspor ke '{config.EXCEL_OUTPUT_FILENAME}'.")
        except Exception as e:
            print(f"[ERROR] Gagal mengekspor data ke Excel: {e}")
        finally:
            print("="*50 + "\n")
    def _load_or_create_vectorstore(self):
        COLLECTION_NAME = "semantic_chunks"
        if os.path.exists(config.DB_NAME):
            print(f"2. Database '{config.DB_NAME}' ditemukan. Memuat...")
            return Chroma(
                persist_directory=config.DB_NAME,
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME
            )
        else:
            print(f"2. Database '{config.DB_NAME}' tidak ditemukan. Membuat baru...")
            if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
                raise RuntimeError(f"Direktori '{config.KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
            pdf_files = [f for f in os.listdir(config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
            if not pdf_files:
                raise RuntimeError(f"Tidak ada PDF di '{config.KNOWLEDGE_BASE_DIR}'.")
            all_final_chunks = []
            for pdf_name in tqdm(pdf_files, desc="Memproses PDF untuk database"):
                pdf_path = os.path.join(config.KNOWLEDGE_BASE_DIR, pdf_name)
                extracted_data = extract_text_with_metadata(pdf_path, config.HEADER_CROP_PERCENTAGE)
                if not extracted_data: continue
                final_chunks = semantic_chunking(
                    extracted_data, 
                    self.embedding_model, 
                    config.SIMILARITY_THRESHOLD,
                    config.MAX_CHARACTERS_PER_CHUNK,
                    config.BATCH_SIZE_EMBEDDING 
                )
                if final_chunks:
                    for chunk in final_chunks: 
                        chunk['metadata']['source_pdf'] = pdf_name
                    all_final_chunks.extend(final_chunks)
            if not all_final_chunks:
                raise RuntimeError("Tidak ada chunk yang berhasil dibuat dari PDF manapun.")
            langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
            print("\n Membuat dan menyimpan database vektor")
            vectorstore = Chroma.from_documents(
                documents=langchain_documents,
                embedding=self.embedding_model,
                persist_directory=config.DB_NAME,
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"} 
            )
            print(f"   > Database berhasil dibuat dengan {vectorstore._collection.count()} dokumen.")
            self._export_database_to_excel(vectorstore)
            return vectorstore
    def _format_chat_history(self):
        if not self.chat_history:
            return "Tidak ada riwayat percakapan."
        return "\n".join([f"Manusia: {q}\nAsisten: {a}" for q, a in self.chat_history])
    def invoke(self, question: str):
        standalone_question = question
        retrieved_results_with_scores = self.retrieve_documents(standalone_question, k=10)
        documents_for_prompt = [
            Document(page_content=result['content'], metadata=result['metadata']) 
            for result in retrieved_results_with_scores
        ]
        document_prompt = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
        context_parts = []
        for doc in documents_for_prompt:
            doc_metadata = {
                "source_range": doc.metadata.get('source_range', 'N/A'),
                "page_content": doc.page_content 
            }
            formatted_doc = document_prompt.format(**doc_metadata)
            context_parts.append(formatted_doc)
        context = "\n".join(context_parts)
        formatted_history = self._format_chat_history()
        rag_chain = self.main_prompt | self.llm
        answer = rag_chain.invoke({
            "chat_history": formatted_history,
            "context": context,
            "question": question
        }).content
        self.chat_history.append((question, answer))
        return {
            "answer": answer,
            "source_documents": retrieved_results_with_scores
        }
    def retrieve_documents(self, question: str, k: int = 10):
        print(f"Melakukan retrieval untuk pertanyaan: '{question}'")
        question_embedding = self.embedding_model.embed_query(question) 
        
        query_results = self.vectorstore._collection.query(
            query_embeddings=[question_embedding],
            n_results=k
        )
        results_list = []
        ids = query_results.get('ids', [[]])[0]
        documents = query_results.get('documents', [[]])[0]
        metadatas = query_results.get('metadatas', [[]])[0]
        distances = query_results.get('distances', [[]])[0] 
        for i in range(len(ids)):
            similarity_score = 1 - distances[i]
            results_list.append({
                "content": documents[i],
                "metadata": metadatas[i],
                "score": similarity_score 
            })
        print(f"   > Ditemukan {len(results_list)} dokumen relevan dengan skor similarity murni.")
        return results_list
def initialize_rag_pipeline():
    return CustomRAGPipeline()