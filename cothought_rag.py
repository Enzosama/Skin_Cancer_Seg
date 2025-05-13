import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()
import torch
from cot_rag import CoTRAG
from rag import QueryParam
from rag.llm import hugging_face_embedding, get_api_key, get_llm_func, get_embedding_func
from PyPDF2 import PdfReader

def load_and_insert_data_cotrag(co_trag, data_path):
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for fname in files:
                path = os.path.join(root, fname)
                ext = fname.lower().rsplit('.', 1)[-1]
                if ext == 'pdf':
                    try:
                        reader = PdfReader(path)
                        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
                    except Exception as e:
                        print(f"Failed to read PDF {path}: {e}", file=sys.stderr)
                        continue
                elif ext in ('txt', 'csv'):
                    with open(path, encoding='utf-8') as f:
                        text = f.read()
                else:
                    continue
                co_trag.insert(text)
    else:
        with open(data_path, encoding='utf-8') as f:
            text = f.read()
        co_trag.insert(text)

async def cotrag_query(question, co_trag, top_k=5):
    param = QueryParam(top_k=top_k)
    res = await co_trag.query(question, param)
    if isinstance(res, (list, tuple)):
        if all(isinstance(x, (float, int)) for x in res):
            if hasattr(res, 'any'):  # Handle numpy arrays
                return "Không thể trả về kết quả dạng vector, vui lòng thử lại với câu hỏi khác."
            return "Không thể trả về kết quả dạng vector, vui lòng thử lại với câu hỏi khác."
        return '\n'.join(str(x) for x in res)
    return str(res) if res else "Không tìm thấy kết quả phù hợp"

def main():
    parser = argparse.ArgumentParser(description="Run CoT-RAG on a Skin Cancer dataset.")
    parser.add_argument("--working_dir", default="./rag_cache")
    parser.add_argument("--data_file", default="Data", help="Path to data file or directory")
    parser.add_argument("--question", required=True)
    parser.add_argument("--engine", choices=["google", "llama", "hugging_face", "openai_github"], default="google")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq", "hugging_face", "openai_github"], default="openai")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    llm_func = get_llm_func(args.engine)
    emb_func = get_embedding_func(args.embed_engine)

    co_trag = CoTRAG(
        working_dir=args.working_dir,
        llm_model_func=llm_func,
        embedding_func=emb_func,
    )

    data_path = args.data_file
    load_and_insert_data_cotrag(co_trag, data_path)

    import asyncio
    async def debug_query():
        res = await cotrag_query(args.question, co_trag, top_k=args.top_k)
        print("[DEBUG] Chain-of-Thought Output:\n", res)
        return res

    result = asyncio.run(debug_query())
    print("[DEBUG] Final COT result:", result)

if __name__ == "__main__":
    main()
