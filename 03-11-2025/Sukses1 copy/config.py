# config.py
KNOWLEDGE_BASE_DIR = "knowledge-base1"
DB_NAME = "Custom_embed"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': True}
SIMILARITY_THRESHOLD = 0.4
HEADER_CROP_PERCENTAGE = 0.15 
MAX_CHARACTERS_PER_CHUNK = 1000 
BATCH_SIZE_EMBEDDING = 32 
EXCEL_OUTPUT_FILENAME = "Custom_embed.xlsx"
DOCUMENT_PROMPT_TEMPLATE = "Rentang: {source_range}\nKonten: {page_content}\n---"
PROMPT_TEMPLATE = """Anda adalah asisten cerdas tanya Alkitab yang mampu menjawab pertanyaan berdasarkan konteks yang diberikan Alkitab dan riwayat percakapan.
Selalu sebutkan sumber informasi dari Alkitab yang kamu gunakan untuk menjawab, jika tidak tau katakan tidak tau.
Jika pertanyaan pengguna adalah tentang percakapan sebelumnya, jawablah berdasarkan riwayat percakapan.
RIWAYAT PERCAKAPAN SEBELUMNYA:
{chat_history}
KONTEKS ALKITAB YANG RELEVAN DENGAN PERTANYAAN SAAT INI:
{context}
PERTANYAAN PENGGUNA SAAT INI:
{question}
Jawaban (ingat sebutkan sumbernya dari KONTEKS ALKITAB jika relevan):"""