import argparse
from rag import RAG, QueryParam
from PathRag.llm import google_complete, llama_complete, openai_embedding, google_embedding, groq_embedding
import os
from PyPDF2 import PdfReader


def main():
    parser = argparse.ArgumentParser(description="Simple RAG runner.")
    parser.add_argument("--working_dir", default="./rag_cache")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--engine", choices=["google", "llama"], default="google")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq"], default="openai")
    args = parser.parse_args()

    # select LLM and embed funcs
    llm_func = google_complete if args.engine == "google" else llama_complete
    if args.embed_engine == "openai":
        emb_func = openai_embedding
    elif args.embed_engine == "google":
        emb_func = google_embedding
    else:
        emb_func = groq_embedding

    rag = RAG(
        working_dir=args.working_dir,
        llm_model_func=llm_func,
        embedding_func=emb_func,
    )
    # Ingest data: support txt, csv, pdf and directory of such files
    data_path = args.data_file
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for fname in files:
                ext = fname.lower().split('.')[-1]
                path = os.path.join(root, fname)
                if ext == 'pdf':
                    try:
                        reader = PdfReader(path)
                        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
                    except Exception as e:
                        print(f"Failed to read PDF {path}: {e}")
                        continue
                elif ext in ('txt', 'csv'):
                    with open(path, encoding='utf-8') as f:
                        text = f.read()
                else:
                    continue
                rag.insert(text)
    else:
        ext = data_path.lower().split('.')[-1]
        if ext == 'pdf':
            reader = PdfReader(data_path)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        else:
            with open(data_path, encoding='utf-8') as f:
                text = f.read()
        rag.insert(text)

    context = "\n\n".join(rag.chunks[i] for i in rag.top_idx)
    system = get_system_prompt()
    prompt = build_rag_prompt(context, args.question, system)
    answer = rag.query(prompt, QueryParam())
    print(answer)
