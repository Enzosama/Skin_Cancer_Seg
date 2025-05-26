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
        import re
        text_stripped = text.strip()
        # Detect CSV by header containing 'question' and 'answer'
        lines = text_stripped.splitlines()
        header_line = lines[0] if lines else ''
        
        # Check if this is the scraped medical content format
        if '=== Document' in text_stripped and 'URL:' in text_stripped and 'Title:' in text_stripped and 'Content:' in text_stripped:
            # Split by document markers
            documents = re.split(r'={10,}\s*\n+\s*=== Document \d+ ===', text_stripped)
            # Remove empty documents
            documents = [doc.strip() for doc in documents if doc.strip()]
            
            new_chunks = []
            new_answers = []
            
            for doc in documents:
                # Extract URL, Title and Content
                url_match = re.search(r'URL:\s*(.*?)\s*\n', doc)
                title_match = re.search(r'Title:\s*(.*?)\s*\n', doc)
                content_match = re.search(r'Content:\s*([\s\S]*?)(?=\n={10,}|$)', doc)
                
                url = url_match.group(1) if url_match else ''
                title = title_match.group(1) if title_match else ''
                content = content_match.group(1).strip() if content_match else ''
                
                if content:
                    # Create a structured chunk with metadata
                    chunk = f"Title: {title}\nURL: {url}\n\n{content}"
                    new_chunks.append(chunk)
                    # Store the content as the answer
                    new_answers.append(content)
        
        elif ',' in header_line and 'question' in header_line.lower() and 'answer' in header_line.lower():
            reader = csv.DictReader(io.StringIO(text))
            new_chunks = []
            new_answers = []
            for row in reader:
                q = row.get('question', '')
                a = row.get('answer', '')
                new_chunks.append(f"{q} {a}".strip())
                new_answers.append(a)
        else:
            # Check if this might be the medical content format without the document markers
            if 'URL:' in text_stripped and 'Title:' in text_stripped and 'Content:' in text_stripped:
                # Try to extract sections
                sections = re.split(r'\n\s*URL:', text_stripped)
                
                new_chunks = []
                new_answers = []
                
                for i, section in enumerate(sections):
                    if i == 0 and not section.strip().startswith('URL:'):
                        # First section might not start with URL if split removed it
                        section = 'URL:' + section
                    
                    # Extract URL, Title and Content
                    url_match = re.search(r'URL:\s*(.*?)\s*\n', section)
                    title_match = re.search(r'Title:\s*(.*?)\s*\n', section)
                    content_match = re.search(r'Content:\s*([\s\S]*?)(?=\n\s*URL:|$)', section)
                    
                    url = url_match.group(1) if url_match else ''
                    title = title_match.group(1) if title_match else ''
                    content = content_match.group(1).strip() if content_match else ''
                    
                    if content:
                        # Create a structured chunk with metadata
                        chunk = f"Title: {title}\nURL: {url}\n\n{content}"
                        new_chunks.append(chunk)
                        # Store the content as the answer
                        new_answers.append(content)
            else:
                # Default text chunking
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
        Enhanced to handle medical content format.
        """
        q_emb_arr = self.embedding_func([question])
        if isinstance(q_emb_arr, np.ndarray):
            q_emb = q_emb_arr[0]
        elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
            q_emb = np.array(q_emb_arr[0])
        else:
            q_emb = np.array(q_emb_arr)

        # Determine number of results from param
        top_k = param.top_k
        top_idx = self.vector_search(q_emb, n_results=top_k)

        if len(top_idx) == 0:
            return "No relevant medical information found in knowledge base."

        # Only use top_k chunks for context and answer
        context_chunks = [self.chunks[i] for i in top_idx]
        
        # Detect if we're dealing with medical content format
        has_medical_content = any(chunk.startswith("Title:") for chunk in context_chunks)
        
        # Format context differently based on content type
        if has_medical_content:
            # Format medical content with source references
            formatted_chunks = []
            for i, chunk in enumerate(context_chunks):
                # Extract title if available
                import re
                title_match = re.search(r'Title:\s*(.*?)\s*\n', chunk)
                title = title_match.group(1) if title_match else f"Source {i+1}"
                
                # Add source reference
                formatted_chunks.append(f"Source {i+1}: {title}\n{chunk}")
            
            context = "\n\n".join(formatted_chunks)
        else:
            context = "\n\n".join(context_chunks)

        # If you want to keep the prompt logic for future LLM use
        from rag.prompt import build_rag_prompt, get_system_prompt
        system = get_system_prompt()
        prompt = build_rag_prompt(context, question, system)

        # Check if this is a medical question
        is_medical_question = any(term in question.lower() for term in [
            "skin", "cancer", "melanoma", "carcinoma", "mole", "nevus", "lesion", 
            "dermatology", "biopsy", "treatment", "diagnosis", "symptom", "abcde"
        ])

        # Format the result based on content type
        if hasattr(self, 'answers') and self.answers:
            if has_medical_content and is_medical_question:
                # For medical content, provide structured response with sources
                result = "Medical Information:\n\n"
                for i, idx in enumerate(top_idx):
                    # Extract title if available
                    chunk = self.chunks[idx]
                    title_match = re.search(r'Title:\s*(.*?)\s*\n', chunk)
                    title = title_match.group(1) if title_match else f"Source {i+1}"
                    
                    result += f"Source {i+1}: {title}\n"
                    result += f"{self.answers[idx]}\n\n"
                
                # Add medical disclaimer
                result += "\nNote: This information is provided for educational purposes only and should not replace professional medical advice."
            else:
                # Standard format for non-medical content
                result = "\n---\n".join(self.answers[i] for i in top_idx)
        else:
            if has_medical_content:
                # Format medical content with source references
                result = "Medical Information:\n\n"
                for i, idx in enumerate(top_idx):
                    chunk = context_chunks[i]
                    title_match = re.search(r'Title:\s*(.*?)\s*\n', chunk)
                    title = title_match.group(1) if title_match else f"Source {i+1}"
                    
                    result += f"Source {i+1}: {title}\n"
                    # Extract content part
                    content_match = re.search(r'\n\n([\s\S]*)', chunk)
                    content = content_match.group(1) if content_match else chunk
                    result += f"{content}\n\n"
            else:
                # Standard format for non-medical content
                result = "\n---\n".join(context_chunks)
        
        return result
