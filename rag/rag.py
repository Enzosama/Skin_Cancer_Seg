import numpy as np
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List

@dataclass
class QueryParam:
    top_k: int = 5

class RAG:
    def __init__(self, working_dir: str, llm_model_func, embedding_func):
        """
        working_dir: path for cache/store (unused in this simple RAG)
        llm_model_func: async function for chat completion
        embedding_func: async function for embeddings
        """
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def insert(self, text: str):
        """
        Ingest a document: chunk by paragraphs and embed all chunks.
        """
        # simple chunk by paragraphs
        self.chunks = [chunk for chunk in text.split('\n\n') if chunk]
        if not self.chunks:
            self.chunks = [text]
        # embed chunks
        self.embeddings = asyncio.run(self.embedding_func(self.chunks))

    def query(self, question: str, param: QueryParam):
        """
        Query the RAG: embed question, find top_k similar chunks, then call LLM with context.
        """
        # embed question
        q_emb_arr = asyncio.run(self.embedding_func([question]))
        # Ensure q_emb is a 1D array
        if isinstance(q_emb_arr, np.ndarray):
            q_emb = q_emb_arr[0]
        elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
            q_emb = np.array(q_emb_arr[0])
        else:
            q_emb = np.array(q_emb_arr)
        # compute cosine similarity
        sims = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        # pick top indices
        top_idx = np.argsort(-sims)[: param.top_k]
        from rag.prompt import build_rag_prompt, get_system_prompt
        context = "\n\n".join(self.chunks[i] for i in top_idx)
        system = get_system_prompt()
        prompt = build_rag_prompt(context, question, system)
        # call LLM
        return asyncio.run(self.llm_model_func(prompt))
