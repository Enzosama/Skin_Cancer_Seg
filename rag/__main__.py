import argparse
from rag import RAG, QueryParam
from PathRag.llm import google_complete, llama_complete, openai_embedding, google_embedding, groq_embedding


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
    with open(args.data_file, encoding="utf-8") as f:
        rag.insert(f.read())
    context = "\n\n".join(rag.chunks[i] for i in rag.top_idx)
    system = get_system_prompt()
    prompt = build_rag_prompt(context, args.question, system)
    answer = rag.query(prompt, QueryParam())
    print(answer)
