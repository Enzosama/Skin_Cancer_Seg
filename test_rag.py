import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()
import torch
from rag import RAG, QueryParam, google_complete, llama_complete, openai_embedding, google_embedding, groq_embedding
from rag.llm import hugging_face_embedding
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#print('DEBUG HUGGING_FACE_API_KEY:', os.environ.get('HUGGING_FACE_API_KEY'))

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

def hugging_face_complete(prompt: str, api_key=None):
    # Dummy function, replace with actual HF LLM if available
    return "[HF LLM] Answer to: " + prompt

async def hugging_face_complete_async(prompt: str, api_key=None):
    return hugging_face_complete(prompt, api_key)

def main():
    parser = argparse.ArgumentParser(description="Run NativeRag on a Skin Cancer dataset.")
    parser.add_argument("--working_dir", default="./rag_cache")
    parser.add_argument("--data_file", default="Data", help="Path to data file or directory")
    parser.add_argument("--question", required=True)
    parser.add_argument("--engine", choices=["google", "llama", "hugging_face"], default="google")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq", "hugging_face"], default="openai")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Select LLM function: luôn trả về nguyên prompt (không wrap, không gọi LLM dummy)
    llm_func = lambda prompt: prompt

    # Select embedding function with API key routing
    if args.embed_engine == "openai":
        api_key = get_api_key("openai")
        emb_func = lambda texts: openai_embedding(texts, api_key=api_key)
    elif args.embed_engine == "hugging_face":
        # hugging_face_embedding does NOT accept api_key, only pass texts
        emb_func = lambda texts: hugging_face_embedding(texts)
    else:
        raise ValueError(f"Unknown embed_engine: {args.embed_engine}")

    rag = RAG(
        working_dir=args.working_dir,
        llm_model_func=llm_func,
        embedding_func=emb_func,
    )

    # Ingest data (file or directory)
    data_path = args.data_file
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for fname in files:
                path = os.path.join(root, fname)
                ext = fname.lower().rsplit('.',1)[-1]
                if ext == 'pdf':
                    try:
                        reader = PdfReader(path)
                        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
                    except Exception as e:
                        print(f"Failed to read PDF {path}: {e}", file=sys.stderr)
                        continue
                    # Chunk PDF into manageable segments
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
                    pdf_chunks = splitter.split_text(text)
                    for chunk in pdf_chunks:
                        print(f"[DEBUG] Loaded PDF chunk ({len(chunk)} chars)")
                        rag.insert(chunk)
                    continue
                elif ext in ('txt','csv'):
                    with open(path, encoding='utf-8') as f:
                        text = f.read()
                    print(f"[DEBUG] Loaded {path} ({len(text)} chars)")
                    rag.insert(text)
                else:
                    continue
    else:
        with open(data_path, encoding="utf-8") as f:
            data = f.read()
            print("[DEBUG] Loaded data (first 300 chars):", data[:300])
            rag.insert(data)
    print("[DEBUG] Chunks after insert:", rag.chunks)
    print("[DEBUG] Embeddings shape:", rag.embeddings.shape)

    # Query and print result
    import asyncio
    async def debug_query():
        print("[DEBUG] Question:", args.question)
        param = QueryParam(top_k=args.top_k)
        res = await rag.query(args.question, param)
        # Split into individual answers
        answers = res.split("\n---\n") if res else []
        print(f"[DEBUG] Returned {len(answers)} answers (expected top_k={param.top_k})")
        for idx, ans in enumerate(answers, 1):
            print(f"Answer {idx}: {ans}")
        return answers
    result = asyncio.run(debug_query())
    print("[DEBUG] Final answers list:", result)


if __name__ == "__main__":
    main()
