import numpy as np
import asyncio
import pickle
import os
import hashlib
import time
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import faiss
from rag.rag import RAG, QueryParam
from rag.llm import hugging_face_embedding

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    cache_dir: str = "./rag_cache"
    sqlite_db_file: str = "faiss_cache.db"
    enable_cache: bool = True
    use_sqlite: bool = True
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
        
        # Initialize SQLite database if enabled
        if self.cache_config.use_sqlite:
            self._init_sqlite_db()
        
        # Load cached data if available
        self._load_cache()
    
    def _get_cache_path(self, filename: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_config.cache_dir, filename)
    
    def _get_db_path(self) -> str:
        """Get full path for SQLite database file"""
        # Store database in the cache directory instead of Data directory
        return os.path.join(self.cache_config.cache_dir, self.cache_config.sqlite_db_file)
    
    def _init_sqlite_db(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self._get_db_path())
            cursor = conn.cursor()
            
            # Create metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL
            )
            ''')
            
            # Create chunks table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                updated_at REAL
            )
            ''')
            
            # Create answers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                updated_at REAL
            )
            ''')
            
            # Create embeddings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB,
                updated_at REAL
            )
            ''')
            
            conn.commit()
            conn.close()
            print(f"[SQLITE] Initialized database at {self._get_db_path()}")
        except Exception as e:
            print(f"[SQLITE] Error initializing database: {e}")
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is not expired"""
        if not self.cache_config.enable_cache:
            return False
        if self.cache_config.use_sqlite:
            try:
                conn = sqlite3.connect(self._get_db_path())
                cursor = conn.cursor()
                cursor.execute("SELECT value, updated_at FROM metadata WHERE key = 'last_update'")
                result = cursor.fetchone()
                conn.close()
                if result:
                    last_update = float(result[1])
                    return time.time() - last_update < self.cache_config.cache_ttl
                return False
            except Exception as e:
                print(f"[SQLITE] Error checking cache validity: {e}")
                return False
        return False
    
    def _load_cache(self):
        """Load cached embeddings, chunks, and FAISS index"""
        try:
            if self.cache_config.use_sqlite:
                self._load_from_sqlite()
            # No file-based cache loading
            if self.embedding_dim is not None and hasattr(self, 'faiss_index'):
                pass  # Optionally, rebuild FAISS index from loaded embeddings
        except Exception as e:
            print(f"[CACHE] Error loading cache: {e}")
            self._reset_cache()
    
    def _load_from_sqlite(self):
        """Load chunks and embeddings from SQLite database"""
        try:
            conn = sqlite3.connect(self._get_db_path())
            cursor = conn.cursor()
            
            # Load chunks
            cursor.execute("SELECT content FROM chunks ORDER BY id")
            self.chunks = [row[0] for row in cursor.fetchall()]
            
            # Load answers
            cursor.execute("SELECT content FROM answers ORDER BY id")
            self.answers = [row[0] for row in cursor.fetchall()]
            
            # Load embeddings
            cursor.execute("SELECT vector FROM embeddings ORDER BY id")
            embedding_blobs = [row[0] for row in cursor.fetchall()]
            
            if embedding_blobs:
                # Convert BLOB data back to numpy arrays
                embeddings_list = [np.frombuffer(blob, dtype=np.float32) for blob in embedding_blobs]
                if embeddings_list:
                    # Reshape to match original dimensions
                    self.embedding_dim = len(embeddings_list[0])
                    self.embeddings = np.vstack(embeddings_list)
                    print(f"[SQLITE] Loaded embeddings with shape {self.embeddings.shape}")
            else:
                self.embeddings = np.array([])
            
            conn.close()
            print(f"[SQLITE] Loaded {len(self.chunks)} chunks from database")
            
        except Exception as e:
            print(f"[SQLITE] Error loading from database: {e}")
            self.chunks = []
            self.embeddings = np.array([])
            self.answers = []
    
    def _load_from_files(self):
        """Load chunks and embeddings from pickle files (legacy method)"""
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
    
    def _save_cache(self):
        """Save embeddings, chunks, and FAISS index to cache"""
        if not self.cache_config.enable_cache:
            return
        try:
            if self.cache_config.use_sqlite:
                self._save_to_sqlite()
            # No file-based cache saving
            print(f"[CACHE] Saved cache with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")
    
    def _save_to_sqlite(self):
        """Save chunks and embeddings to SQLite database"""
        try:
            conn = sqlite3.connect(self._get_db_path())
            cursor = conn.cursor()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Clear existing data
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM answers")
            cursor.execute("DELETE FROM embeddings")
            
            # Save chunks
            current_time = time.time()
            for chunk in self.chunks:
                cursor.execute("INSERT INTO chunks (content, updated_at) VALUES (?, ?)", 
                              (chunk, current_time))
            
            # Save answers
            for answer in self.answers:
                cursor.execute("INSERT INTO answers (content, updated_at) VALUES (?, ?)", 
                              (answer, current_time))
            
            # Save embeddings
            if len(self.embeddings) > 0:
                for i in range(len(self.embeddings)):
                    # Convert numpy array to binary blob
                    vector_blob = self.embeddings[i].astype(np.float32).tobytes()
                    cursor.execute("INSERT INTO embeddings (vector, updated_at) VALUES (?, ?)", 
                                  (vector_blob, current_time))
            
            # Update metadata
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
                          ("last_update", str(current_time), current_time))
            
            # Commit transaction
            conn.commit()
            conn.close()
            print(f"[SQLITE] Saved {len(self.chunks)} chunks to database")
            
        except Exception as e:
            print(f"[SQLITE] Error saving to database: {e}")
    
    def _save_to_files(self):
        """Save chunks and embeddings to pickle files (legacy method)"""
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
    
    def _reset_cache(self):
        """Reset all cached data"""
        self.chunks = []
        self.embeddings = np.array([])
        self.answers = []
        self.faiss_index = None
        self.embedding_dim = None
        
        if self.cache_config.use_sqlite:
            try:
                conn = sqlite3.connect(self._get_db_path())
                cursor = conn.cursor()
                
                # Clear all tables
                cursor.execute("DELETE FROM chunks")
                cursor.execute("DELETE FROM answers")
                cursor.execute("DELETE FROM embeddings")
                cursor.execute("DELETE FROM metadata")
                
                conn.commit()
                conn.close()
                print("[SQLITE] Reset database cache")
            except Exception as e:
                print(f"[SQLITE] Error resetting database: {e}")
    
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
        stats = {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_dim,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'query_cache_size': len(self.query_cache),
            'cache_enabled': self.cache_config.enable_cache
        }
        
        if self.cache_config.use_sqlite:
            try:
                conn = sqlite3.connect(self._get_db_path())
                cursor = conn.cursor()
                # Get database size
                cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                # Get table counts
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunks_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                embeddings_count = cursor.fetchone()[0]
                conn.close()
                stats.update({
                    'sqlite_db_size': db_size,
                    'sqlite_chunks_count': chunks_count,
                    'sqlite_embeddings_count': embeddings_count
                })
            except Exception as e:
                print(f"[SQLITE] Error getting stats: {e}")
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache = {}
        if self.cache_config.use_sqlite:
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
                        enable_cache: bool = True,
                        use_sqlite: bool = True) -> OptimizedRAG:
    """Create an optimized RAG instance with default settings"""
    if embedding_func is None:
        embedding_func = hugging_face_embedding
    if llm_model_func is None:
        llm_model_func = lambda prompt: prompt  # Default passthrough
    cache_config = CacheConfig(
        cache_dir=working_dir,
        enable_cache=enable_cache,
        use_sqlite=use_sqlite
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

def faiss_index():
    """Preload FAISS index at app startup for shared use."""
    rag = create_optimized_rag(
        working_dir="./rag_cache",
        llm_model_func=None,
        embedding_func=hugging_face_embedding,
        enable_cache=True,
        use_sqlite=True
    )
    # This will trigger FAISS index loading if available
    if rag.faiss_index is not None:
        print(f"[PRELOAD] FAISS index loaded with {rag.faiss_index.ntotal} vectors.")
    else:
        print("[PRELOAD] FAISS index not found or empty.")
    return rag