# config.py
KNOWLEDGE_BASE_DIR = "knowledge-base"
# DB_NAME = "chroma_db_normal_semantic"
DB_NAME = "DB_0.4"
# DB_NAME = "testing7"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}
# ENCODE_KWARGS = {'normalize_embeddings': True}
SIMILARITY_THRESHOLD = 0.4
DOCUMENT_PROMPT_TEMPLATE = "Rentang: {source_range}\nKonten: {page_content}\n---"
PROMPT_TEMPLATE = """Anda adalah asisten cerdas tanya Alkitab yang mampu menjawab pertanyaan berdasarkan konteks yang diberikan Alkitab.
Selalu sebutkan sumber informasi yang kamu gunakan untuk  menjawab, jika tidak tau katakan tidak tau.
Konteks:
{context}
Pertanyaan:
{question}
Jawaban (ingat sebutkan sumbernya):"""