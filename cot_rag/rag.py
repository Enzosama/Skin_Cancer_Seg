import numpy as np
import asyncio
from rag.rag import RAG, QueryParam
from rag.prompt import get_system_prompt

class CoTRAG(RAG):
    """
    Chain-of-Thought RAG: extends RAG to include step-by-step reasoning in answers.
    """

    async def query(self, question: str, param: QueryParam = QueryParam()):
        """
        Query the Chain-of-Thought RAG: embed question, retrieve context,
        then prompt the LLM with chain-of-thought instructions.
        """
        # Embed question
        q_emb_arr = self.embedding_func([question])
        if isinstance(q_emb_arr, np.ndarray):
            q_emb = q_emb_arr[0]
        elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
            q_emb = np.array(q_emb_arr[0])
        else:
            q_emb = np.array(q_emb_arr)

        # Retrieve top-k chunks
        top_k = param.top_k
        top_idx = self.vector_search(q_emb, n_results=top_k)
        if len(top_idx) == 0:
            return "No relevant answer found in knowledge base."

        # Build context
        context_chunks = [self.chunks[i] for i in top_idx]
        context = "\n\n".join(context_chunks)

        # Prepare chain-of-thought prompt
        system_prompt = get_system_prompt()
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Please think through this step by step (chain of thought), then provide a final concise answer."
        )

        # Call LLM
        if asyncio.iscoroutinefunction(self.llm_model_func):
            result = await self.llm_model_func(prompt)
        else:
            # support sync functions
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.llm_model_func, prompt)
        return result
