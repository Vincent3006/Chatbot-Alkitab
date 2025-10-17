# rag_pipeline.py

import os
from tqdm import tqdm

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import config
from utils import extract_text_with_metadata, semantic_chunking

def initialize_rag_pipeline():
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.MODEL_NAME,
        model_kwargs=config.MODEL_KWARGS,
        encode_kwargs=config.ENCODE_KWARGS
    )

    if os.path.exists(config.DB_NAME):
        print(f"2. Database '{config.DB_NAME}' ditemukan. Memuat...")
        vectorstore = Chroma(persist_directory=config.DB_NAME, embedding_function=embedding_model)
    else:
        print(f"2. Database '{config.DB_NAME}' tidak ditemukan. Membuat baru...")
        if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
            raise RuntimeError(f"Direktori '{config.KNOWLEDGE_BASE_DIR}' tidak ditemukan. Startup Gagal.")
        
        pdf_files = [f for f in os.listdir(config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            raise RuntimeError(f"Tidak ada PDF di '{config.KNOWLEDGE_BASE_DIR}'. Startup Gagal.")

        all_final_chunks = []
        for pdf_name in tqdm(pdf_files, desc="Memproses PDF untuk database"):
            pdf_path = os.path.join(config.KNOWLEDGE_BASE_DIR, pdf_name)
            extracted_data = extract_text_with_metadata(pdf_path)
            if not extracted_data: continue

            final_chunks = semantic_chunking(extracted_data, embedding_model, config.SIMILARITY_THRESHOLD)
            if final_chunks:
                for chunk in final_chunks:
                    chunk['metadata']['source_pdf'] = pdf_name 
                all_final_chunks.extend(final_chunks)

        if not all_final_chunks:
            raise RuntimeError("Tidak ada chunk yang berhasil dibuat. Startup Gagal.")

        langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
        vectorstore = Chroma.from_documents(documents=langchain_documents, embedding=embedding_model, persist_directory=config.DB_NAME)

    print(f"Jumlah dokumen: {vectorstore._collection.count()}")

    # if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    #     raise RuntimeError("GOOGLE_API_KEY tidak ditemukan. Startup gagal.")
    # else:
    #     print("4. GOOGLE_API_KEY ditemukan.")


    DOCUMENT_PROMPT = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
    PROMPT = PromptTemplate(template=config.PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT, "document_prompt": DOCUMENT_PROMPT},
        verbose=True
    )
    
    return conversation_chain