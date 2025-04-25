import argparse
import os
import sys
from rag import RAG, QueryParam, google_complete, llama_complete, openai_embedding, google_embedding, groq_embedding
from rag.llm import hugging_face_embedding
import os

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

def main():
    parser = argparse.ArgumentParser(description="Run NativeRag on a Skin Cancer dataset.")
    parser.add_argument("--working_dir", default="./rag_cache")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--engine", choices=["google", "llama", "hugging_face"], default="google")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq", "hugging_face"], default="openai")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Select LLM function with API key routing
    if args.engine == "google":
        api_key = get_api_key("google")
        llm_func = lambda prompt: google_complete(prompt, api_key=api_key)
    elif args.engine == "llama":
        llm_func = llama_complete  # Local, no API key needed
    elif args.engine == "hugging_face":
        api_key = get_api_key("hugging_face")
        llm_func = lambda prompt: hugging_face_complete(prompt, api_key=api_key)
    else:
        raise ValueError(f"Unknown engine: {args.engine}")

    # Select embedding function with API key routing
    if args.embed_engine == "openai":
        api_key = get_api_key("openai")
        emb_func = lambda texts: openai_embedding(texts, api_key=api_key)
    elif args.embed_engine == "hugging_face":
        api_key = get_api_key("hugging_face")
        emb_func = lambda texts: hugging_face_embedding(texts, api_key=api_key)
    else:
        raise ValueError(f"Unknown embed_engine: {args.embed_engine}")

    rag = RAG(
        working_dir=args.working_dir,
        llm_model_func=llm_func,
        embedding_func=emb_func,
    )

    # Ingest data
    with open(args.data_file, encoding="utf-8") as f:
        rag.insert(f.read())

    # Query and print result
    result = rag.query(args.question, QueryParam(top_k=args.top_k))
    print(result)

if __name__ == "__main__":
    main()
