import numpy as np
import asyncio
import pickle
import os
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import faiss
from rag.rag import RAG, QueryParam
from rag.llm import hugging_face_embedding

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    cache_dir: str = "./rag_cache"
    embeddings_cache_file: str = "embeddings.pkl"
    chunks_cache_file: str = "chunks.pkl"
    index_cache_file: str = "faiss_index.bin"
    enable_cache: bool = True
    cache_ttl: int = 86400  # 24 hours 

class OptimizedRAG(RAG):   
    def __init__(self, working_dir: str, llm_model_func, embedding_func, cache_config: Optional[CacheConfig] = None):
        super().__init__(working_dir, llm_model_func, embedding_func)
        self.cache_config = cache_config or CacheConfig(cache_dir=working_dir)
        self.faiss_index = None
        self.embedding_dim = None
        self.query_cache = {}  # In-memory query cache
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure cache directory exists
        os.makedirs(self.cache_config.cache_dir, exist_ok=True)
        
        # Load cached data if available
        self._load_cache()
    
    def _get_cache_path(self, filename: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_config.cache_dir, filename)
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is not expired"""
        if not self.cache_config.enable_cache:
            return False
        
        cache_path = self._get_cache_path(cache_file)
        if not os.path.exists(cache_path):
            return False
        
        # Check TTL
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < self.cache_config.cache_ttl
    
    def _load_cache(self):
        """Load cached embeddings, chunks, and FAISS index"""
        try:
            # Load chunks
            if self._is_cache_valid(self.cache_config.chunks_cache_file):
                with open(self._get_cache_path(self.cache_config.chunks_cache_file), 'rb') as f:
                    cache_data = pickle.load(f)
                    self.chunks = cache_data.get('chunks', [])
                    self.answers = cache_data.get('answers', [])
                    print(f"[CACHE] Loaded {len(self.chunks)} chunks from cache")
            
            # Load embeddings
            if self._is_cache_valid(self.cache_config.embeddings_cache_file):
                with open(self._get_cache_path(self.cache_config.embeddings_cache_file), 'rb') as f:
                    self.embeddings = pickle.load(f)
                    if len(self.embeddings) > 0:
                        self.embedding_dim = self.embeddings.shape[1]
                    print(f"[CACHE] Loaded embeddings with shape {self.embeddings.shape}")
            
            # Load FAISS index
            if (self._is_cache_valid(self.cache_config.index_cache_file) and 
                self.embedding_dim is not None):
                index_path = self._get_cache_path(self.cache_config.index_cache_file)
                self.faiss_index = faiss.read_index(index_path)
                print(f"[CACHE] Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                
        except Exception as e:
            print(f"[CACHE] Error loading cache: {e}")
            self._reset_cache()
    
    def _save_cache(self):
        """Save embeddings, chunks, and FAISS index to cache"""
        if not self.cache_config.enable_cache:
            return
        
        try:
            # Save chunks and answers
            cache_data = {
                'chunks': self.chunks,
                'answers': self.answers
            }
            with open(self._get_cache_path(self.cache_config.chunks_cache_file), 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save embeddings
            if len(self.embeddings) > 0:
                with open(self._get_cache_path(self.cache_config.embeddings_cache_file), 'wb') as f:
                    pickle.dump(self.embeddings, f)
            
            # Save FAISS index
            if self.faiss_index is not None:
                index_path = self._get_cache_path(self.cache_config.index_cache_file)
                faiss.write_index(self.faiss_index, index_path)
            
            print(f"[CACHE] Saved cache with {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")
    
    def _reset_cache(self):
        """Reset all cached data"""
        self.chunks = []
        self.embeddings = np.array([])
        self.answers = []
        self.faiss_index = None
        self.embedding_dim = None
        self.query_cache = {}
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if len(self.embeddings) == 0:
            return
        
        if self.embedding_dim is None:
            self.embedding_dim = self.embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # Add to index
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
        print(f"[FAISS] Built index with {self.faiss_index.ntotal} vectors")
    
    def insert(self, text: str, batch_size: int = 32):
        """Insert text with optimized batch processing"""
        import csv, io
        text_stripped = text.strip()
        
        # Detect CSV format
        lines = text_stripped.splitlines()
        header_line = lines[0] if lines else ''
        
        if ',' in header_line and 'question' in header_line.lower() and 'answer' in header_line.lower():
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
        
        # Process in batches for memory efficiency
        all_embeddings = []
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            print(f"[BATCH] Processing batch {i//batch_size + 1}/{(len(new_chunks) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for batch
            batch_emb = self.embedding_func(batch)
            if not isinstance(batch_emb, np.ndarray):
                batch_emb = np.array(batch_emb)
            
            all_embeddings.append(batch_emb)
        
        # Combine all embeddings
        if all_embeddings:
            new_emb = np.vstack(all_embeddings)
        else:
            return
        
        # Update data structures
        if self.embeddings.size == 0:
            self.chunks = new_chunks
            self.embeddings = new_emb
            self.answers = new_answers
            self.embedding_dim = new_emb.shape[1]
        else:
            self.chunks.extend(new_chunks)
            self.embeddings = np.vstack([self.embeddings, new_emb])
            self.answers.extend(new_answers)
        
        # Rebuild FAISS index
        self._build_faiss_index()
        
        # Save to cache
        self._save_cache()
        
        print(f"[INSERT] Added {len(new_chunks)} chunks. Total: {len(self.chunks)}")
    
    def vector_search(self, query_embedding, n_results=3) -> List[int]:
        """Optimized vector search using FAISS"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_emb = query_embedding.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_emb)
        
        # Search using FAISS
        n_results = min(n_results, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_emb, n_results)
        
        return indices[0].tolist()
    
    def _get_query_hash(self, question: str, param: QueryParam) -> str:
        """Generate hash for query caching"""
        query_str = f"{question}_{param.top_k}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def query(self, question: str, param: QueryParam = QueryParam()):
        """Optimized query with caching and parallel processing"""
        query_hash = self._get_query_hash(question, param)
        if query_hash in self.query_cache:
            print(f"[CACHE] Query cache hit")
            return self.query_cache[query_hash]
        
        # Generate embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        q_emb_arr = await loop.run_in_executor(
            self.executor, 
            self.embedding_func, 
            [question]
        )
        
        if isinstance(q_emb_arr, np.ndarray):
            q_emb = q_emb_arr[0]
        elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
            q_emb = np.array(q_emb_arr[0])
        else:
            q_emb = np.array(q_emb_arr)
        
        # Vector search
        top_k = param.top_k
        top_idx = self.vector_search(q_emb, n_results=top_k)
        
        if len(top_idx) == 0:
            result = "No relevant answer found in knowledge base."
        else:
            # Prepare result
            if hasattr(self, 'answers') and self.answers:
                result = "\n---\n".join(self.answers[i] for i in top_idx if i < len(self.answers))
            else:
                context_chunks = [self.chunks[i] for i in top_idx if i < len(self.chunks)]
                result = "\n---\n".join(context_chunks)
        
        # Cache result
        self.query_cache[query_hash] = result
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.query_cache.keys())[:100]
            for key in oldest_keys:
                del self.query_cache[key]
        
        return result
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_dim,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'query_cache_size': len(self.query_cache),
            'cache_enabled': self.cache_config.enable_cache
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache = {}
        
        # Remove cache files
        cache_files = [
            self.cache_config.chunks_cache_file,
            self.cache_config.embeddings_cache_file,
            self.cache_config.index_cache_file
        ]
        
        for cache_file in cache_files:
            cache_path = self._get_cache_path(cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"[CACHE] Removed {cache_path}")
        
        self._reset_cache()
        print("[CACHE] All caches cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Factory function for easy integration
def create_optimized_rag(working_dir: str = "./rag_cache", 
                        llm_model_func=None, 
                        embedding_func=None,
                        enable_cache: bool = True) -> OptimizedRAG:
    """Create an optimized RAG instance with default settings"""
    
    if embedding_func is None:
        embedding_func = hugging_face_embedding
    
    if llm_model_func is None:
        llm_model_func = lambda prompt: prompt  # Default passthrough
    
    cache_config = CacheConfig(
        cache_dir=working_dir,
        enable_cache=enable_cache
    )
    
    return OptimizedRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        cache_config=cache_config
    )

# Utility function for benchmarking
async def benchmark_rag(rag_instance, test_queries: List[str], param: QueryParam = QueryParam()) -> Dict:
    """Benchmark RAG performance"""
    import time
    
    results = {
        'total_queries': len(test_queries),
        'total_time': 0,
        'avg_time_per_query': 0,
        'queries_per_second': 0,
        'individual_times': []
    }
    
    start_time = time.time()
    
    for i, query in enumerate(test_queries):
        query_start = time.time()
        await rag_instance.query(query, param)
        query_time = time.time() - query_start
        results['individual_times'].append(query_time)
        
        if (i + 1) % 10 == 0:
            print(f"[BENCHMARK] Processed {i + 1}/{len(test_queries)} queries")
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['avg_time_per_query'] = total_time / len(test_queries)
    results['queries_per_second'] = len(test_queries) / total_time
    
    return results