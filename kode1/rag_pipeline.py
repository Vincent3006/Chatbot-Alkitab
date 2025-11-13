#kode1\rag_pipeline.py
import os
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI 
import config
from utils import extract_text_with_metadata, semantic_chunking
CONDENSE_QUESTION_PROMPT_TEMPLATE = """
Mengingat riwayat percakapan dan pertanyaan tindak lanjut, ubah pertanyaan tindak lanjut tersebut menjadi pertanyaan yang berdiri sendiri.
Riwayat Percakapan:
{chat_history}
Pertanyaan Tindak Lanjut:
{question}
Pertanyaan yang Berdiri Sendiri:"""
class CustomRAGPipeline:
    def __init__(self):
        print("1. Menginisialisasi Model dan Database...")
        self.embedding_model = self._initialize_embeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.llm = self._initialize_llm()
        self.chat_history = []
        self.main_prompt = PromptTemplate(
            template=config.PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        self.condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)
    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(
        model_name=config.MODEL_NAME,
        model_kwargs=config.MODEL_KWARGS,
        encode_kwargs=config.ENCODE_KWARGS 
    )
    def _initialize_llm(self):
        return ChatOllama(
            model="qwen2.5:3b",
            temperature=0.3
        )
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
                extracted_data = extract_text_with_metadata(pdf_path)
                if not extracted_data: continue
                final_chunks = semantic_chunking(extracted_data, self.embedding_model, config.SIMILARITY_THRESHOLD)
                if final_chunks:
                    for chunk in final_chunks: 
                        chunk['metadata']['source_pdf'] = pdf_name
                    all_final_chunks.extend(final_chunks)
            if not all_final_chunks:
                raise RuntimeError("Tidak ada chunk yang berhasil dibuat.")
            langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
            vectorstore = Chroma.from_documents(
                documents=langchain_documents,
                embedding=self.embedding_model,
                persist_directory=config.DB_NAME,
                collection_name=COLLECTION_NAME
            )
            print(f"Jumlah dokumen: {vectorstore._collection.count()}")
            return vectorstore
    def _format_chat_history(self):
        """Mengubah riwayat chat menjadi string."""
        return "\n".join([f"Manusia: {q}\nAsisten: {a}" for q, a in self.chat_history])
    def invoke(self, question: str):
        """Fungsi utama untuk memproses permintaan chat."""
        standalone_question = question
        if self.chat_history:
            print("   > Mengondensasi pertanyaan dengan riwayat chat...")
            condense_chain = self.condense_question_prompt | self.llm
            standalone_question = condense_chain.invoke({
                "chat_history": self._format_chat_history(),
                "question": question
            }).content
        print(f"   > Pertanyaan yang akan dicari: '{standalone_question}'")
        print("   > Melakukan retrieval dokumen...")
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(standalone_question, k=10)
        print("\n" + "="*50)
        print("--- HASIL RETRIEVAL DARI DATABASE ---")
        print("="*50 + "\n")
        print("   > Memformat konteks untuk LLM...")
        document_prompt = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
        context_parts = []
        for doc, score in retrieved_docs_with_scores:
            doc_metadata = {
                "source_range": doc.metadata.get('source_range', 'N/A'),
                "page_content": doc.page_content 
            }
            formatted_doc = document_prompt.format(**doc_metadata)
            context_parts.append(formatted_doc)
        context = "\n".join(context_parts)
        final_prompt_for_llm = self.main_prompt.format(context=context, question=question)
        print("\n" + "="*50)
        print("--- PROMPT FINAL YANG DIKIRIM KE LLM ---")
        print("="*50 + "\n")
        print("   > Menghasilkan jawaban akhir...")
        rag_chain = self.main_prompt | self.llm
        answer = rag_chain.invoke({
            "context": context,
            "question": question
        }).content
        self.chat_history.append((question, answer))
        return {
            "answer": answer,
            "source_documents_with_scores": retrieved_docs_with_scores  
        }
    def retrieve_documents(self, question: str, k: int = 4):
        print(f"   > Melakukan retrieval untuk pertanyaan: '{question}'")
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        results = []
        for doc, score in retrieved_docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        print(f"   > Ditemukan {len(results)} dokumen relevan.")
        return results
def initialize_rag_pipeline():
    """Fungsi pembungkus untuk membuat instance pipeline."""
    return CustomRAGPipeline()