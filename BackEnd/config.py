# config.py

KNOWLEDGE_BASE_DIR = "knowledge-base"
DB_NAME = "chroma_db_Semantic2"

# -Embedding 
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}


SIMILARITY_THRESHOLD = 0.7

DOCUMENT_PROMPT_TEMPLATE = "Rentang: {source_range}\nKonten: {page_content}\n---"

PROMPT_TEMPLATE = """Anda adalah asisten cerdas yang menjawab pertanyaan berdasarkan konteks yang diberikan.
Selalu sebutkan sumber informasi (nama PDF dan rentangnya) yang kamu gunakan untuk menjawab.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban (ingat sebutkan sumbernya):"""