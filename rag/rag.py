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
        embedding_func: function for embeddings
        """
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        # Store answers separately when ingesting CSV
        self.answers: List[str] = []

    def insert(self, text: str):
        import csv, io
        text_stripped = text.strip()
        # Chunk text and parse CSV answers
        if text_stripped.startswith('type_disease'):
            reader = csv.DictReader(io.StringIO(text))
            new_chunks = []
            new_answers = []
            for row in reader:
                q = row.get('question', '')
                a = row.get('answer', '')
                new_chunks.append(f"{q} {a}".strip())
                new_answers.append(a)
        else:
            new_chunks = [chunk for chunk in text.split('\n\n') if chunk]
            if not new_chunks:
                new_chunks = [text]
            new_answers = []
        # Embed new chunks
        new_emb = self.embedding_func(new_chunks)
        if not isinstance(new_emb, np.ndarray):
            new_emb = np.array(new_emb)
        # Append or initialize
        if self.embeddings.size == 0:
            self.chunks = new_chunks
            self.embeddings = new_emb
            self.answers = new_answers
        else:
            self.chunks.extend(new_chunks)
            self.embeddings = np.vstack([self.embeddings, new_emb])
            self.answers.extend(new_answers)

    def vector_search(self, query_embedding, n_results=3):
        """
        Perform vector search over stored embeddings.
        Returns indices of top n_results most similar chunks.
        """
        if len(self.embeddings) == 0:
            return []
        sims = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_idx = np.argsort(-sims)[:n_results]
        return top_idx

    async def query(self, question: str, param: QueryParam = QueryParam()):
        """
        Query the RAG: embed question, find top_k similar chunks, then return answers.
        """
        q_emb_arr = self.embedding_func([question])
        if isinstance(q_emb_arr, np.ndarray):
            q_emb = q_emb_arr[0]
        elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
            q_emb = np.array(q_emb_arr[0])
        else:
            q_emb = np.array(q_emb_arr)

        # Always search for at most 3 results
        top_k = min(3, param.top_k)
        top_idx = self.vector_search(q_emb, n_results=top_k)
        print("[DEBUG][RAG.query] Top idx:", top_idx)

        if len(top_idx) == 0:
            return "No relevant answer found in knowledge base."

        # Only use top_k chunks for context and answer
        context_chunks = [self.chunks[i] for i in top_idx]
        context = "\n\n".join(context_chunks)
        print("[DEBUG][RAG.query] Context:", context)

        # If you want to keep the prompt logic for future LLM use
        from rag.prompt import build_rag_prompt, get_system_prompt
        system = get_system_prompt()
        prompt = build_rag_prompt(context, question, system)
        print("[DEBUG][RAG.query] Prompt:", prompt)

        # Return up to 3 answers: only the answer texts if available
        if hasattr(self, 'answers') and self.answers:
            result = "\n---\n".join(self.answers[i] for i in top_idx)
        else:
            result = "\n---\n".join(context_chunks)
        return result
