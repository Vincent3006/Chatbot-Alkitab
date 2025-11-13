# debug_startup.py
print("debug...")
try:
    from rag_pipeline import initialize_rag_pipeline
    
    chain = initialize_rag_pipeline()
    if chain:
        print("RAG Chain berhasil")
    else:
        print("Terjadi Error")
        
except Exception as e:
    import traceback
    print("Error saat inisialisasi:", e)
    traceback.print_exc()
