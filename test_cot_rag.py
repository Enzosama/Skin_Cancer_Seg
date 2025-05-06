import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()
import torch
from cot_rag import CoTRAG
from rag import QueryParam
from rag.llm import hugging_face_embedding
from PyPDF2 import PdfReader

# API key loader
def get_api_key(service: str):
    env_map = {
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai_github": "OPENAI_GITHUB_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "hugging_face": "HUGGING_FACE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key = env_map.get(service)
    if not key:
        raise ValueError(f"No API key mapping for service {service}")
    value = os.environ.get(key)
    if not value:
        print(f"Missing API key for {service} ({key})", file=sys.stderr)
        sys.exit(1)
    return value

# Dummy HF LLM
def hugging_face_complete(prompt: str, api_key=None):
    return "[HF LLM] Answer to: " + prompt

async def hugging_face_complete_async(prompt: str, api_key=None):
    return hugging_face_complete(prompt, api_key)


def main():
    parser = argparse.ArgumentParser(description="Run CoT-RAG on a Skin Cancer dataset.")
    parser.add_argument("--working_dir", default="./rag_cache")
    parser.add_argument("--data_file", default="Data", help="Path to data file or directory")
    parser.add_argument("--question", required=True)
    parser.add_argument("--engine", choices=["google", "llama", "hugging_face"], default="google")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq", "hugging_face"], default="openai")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # always return prompt for testing
    llm_func = lambda prompt: prompt

    # select embed function
    if args.embed_engine == "openai":
        api_key = get_api_key("openai")
        emb_func = lambda texts: __import__('rag').rag.openai_embedding(texts, api_key=api_key)
    elif args.embed_engine == "hugging_face":
        emb_func = lambda texts: hugging_face_embedding(texts)
    elif args.embed_engine == "google":
        api_key = get_api_key("google")
        emb_func = lambda texts: __import__('rag').rag.google_embedding(texts, api_key=api_key)
    elif args.embed_engine == "groq":
        api_key = get_api_key("groq")
        emb_func = lambda texts: __import__('rag').rag.groq_embedding(texts, api_key=api_key)
    else:
        raise ValueError(f"Unknown embed_engine: {args.embed_engine}")

    co_trag = CoTRAG(
        working_dir=args.working_dir,
        llm_model_func=llm_func,
        embedding_func=emb_func,
    )

    data_path = args.data_file
    # ingest data
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

    # run query with chain-of-thought
    import asyncio
    async def debug_query():
        param = QueryParam(top_k=args.top_k)
        res = await co_trag.query(args.question, param)
        print("[DEBUG] Chain-of-Thought Output:\n", res)
        return res

    result = asyncio.run(debug_query())
    print("[DEBUG] Final COT result:", result)


if __name__ == "__main__":
    main()
