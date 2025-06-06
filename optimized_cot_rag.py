import numpy as np
import asyncio
import pickle
import os
import hashlib
import time
import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import faiss
from optimized_rag import OptimizedRAG, CacheConfig
from cot_rag.rag import CoTStep, CoTResult
from rag.rag import QueryParam
from rag.prompt import get_system_prompt

@dataclass
class CoTCacheConfig(CacheConfig):
    """Extended cache configuration for CoT-RAG"""
    reasoning_cache_file: str = "reasoning_cache.pkl"
    cot_results_cache_file: str = "cot_results.pkl"
    enable_reasoning_cache: bool = True

class OptimizedCoTRAG(OptimizedRAG):
    """
    Optimized Chain-of-Thought RAG with:
    - All OptimizedRAG features (FAISS, caching, batch processing)
    - Cached reasoning steps and patterns
    - Parallel reasoning step processing
    - Optimized prompt generation
    - Smart reasoning step reuse
    """
    
    def __init__(self, working_dir: str, llm_model_func, embedding_func, 
                 max_reasoning_steps: int = 5, confidence_threshold: float = 0.7,
                 cache_config: Optional[CoTCacheConfig] = None):
        
        # Initialize with CoT-specific cache config
        if cache_config is None:
            cache_config = CoTCacheConfig(cache_dir=working_dir)
        
        super().__init__(working_dir, llm_model_func, embedding_func, cache_config)
        
        self.max_reasoning_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Additional CoT-specific attributes
        self.reasoning_cache = {}
        self.cot_results_cache = {}
        self.cot_results = []
        
        # Load CoT-specific cached data
        self._load_cot_cache()
    
    def _get_db_path(self) -> str:
        """Get full path for SQLite database file"""
        # Store database in the cache directory instead of Data directory
        return os.path.join(self.cache_config.cache_dir, self.cache_config.sqlite_db_file)
    
    def __del__(self):
        """Save CoT caches before destruction"""
        if hasattr(self, 'cache_config') and self.cache_config.enable_cache:
            try:
                self._save_cot_cache()
            except Exception as e:
                print(f"[CACHE] Error saving CoT cache during cleanup: {e}")
            super().__del__()

    def _load_cot_cache(self):
        """Load CoT-specific cached data"""
        try:
            if self.cache_config.use_sqlite:
                self._load_cot_from_sqlite()
            else:
                self._load_cot_from_files()
        except Exception as e:
            print(f"[COT_CACHE] Error loading CoT cache: {e}")
            self.reasoning_cache = {}
            self.cot_results_cache = {}
    
    def _load_cot_from_sqlite(self):
        """Load CoT-specific caches from SQLite database"""
        try:
            conn = sqlite3.connect(self._get_db_path())
            cursor = conn.cursor()
            
            # Check if CoT tables exist, create them if not
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                query_hash TEXT PRIMARY KEY,
                data BLOB,
                updated_at REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cot_results_cache (
                query_hash TEXT PRIMARY KEY,
                data BLOB,
                updated_at REAL
            )
            ''')
            
            # Load reasoning cache
            cursor.execute("SELECT query_hash, data FROM reasoning_cache")
            for row in cursor.fetchall():
                query_hash, data_blob = row
                self.reasoning_cache[query_hash] = pickle.loads(data_blob)
            
            # Load CoT results cache
            cursor.execute("SELECT query_hash, data FROM cot_results_cache")
            for row in cursor.fetchall():
                query_hash, data_blob = row
                self.cot_results_cache[query_hash] = pickle.loads(data_blob)
            
            conn.close()
            print(f"[SQLITE] Loaded {len(self.reasoning_cache)} reasoning patterns and {len(self.cot_results_cache)} CoT results")
            
        except Exception as e:
            print(f"[SQLITE] Error loading CoT data from database: {e}")
            self.reasoning_cache = {}
            self.cot_results_cache = {}
    
    def _load_cot_from_files(self):
        """Load CoT-specific caches from pickle files (legacy method)"""
        # Load reasoning cache
        reasoning_cache_path = self._get_cache_path(self.cache_config.reasoning_cache_file)
        if self._is_cache_valid(self.cache_config.reasoning_cache_file) and os.path.exists(reasoning_cache_path):
            with open(reasoning_cache_path, 'rb') as f:
                self.reasoning_cache = pickle.load(f)
                print(f"[CACHE] Loaded {len(self.reasoning_cache)} reasoning patterns from cache")
            
        # Load CoT results cache
        cot_results_cache_path = self._get_cache_path(self.cache_config.cot_results_cache_file)
        if self._is_cache_valid(self.cache_config.cot_results_cache_file) and os.path.exists(cot_results_cache_path):
            with open(cot_results_cache_path, 'rb') as f:
                self.cot_results_cache = pickle.load(f)
                print(f"[CACHE] Loaded {len(self.cot_results_cache)} CoT results from cache")
    
    def _save_cot_cache(self):
        """Save CoT-specific cached data"""
        if not self.cache_config.enable_cache:
            return
        
        try:
            if self.cache_config.use_sqlite:
                self._save_cot_to_sqlite()
            else:
                self._save_cot_to_files()
            
            print(f"[CACHE] Saved CoT caches")
            
        except Exception as e:
            print(f"[COT_CACHE] Error saving CoT cache: {e}")
    
    def _save_cot_to_sqlite(self):
        """Save CoT-specific caches to SQLite database"""
        try:
            conn = sqlite3.connect(self._get_db_path())
            cursor = conn.cursor()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Save reasoning cache
            current_time = time.time()
            for query_hash, data in self.reasoning_cache.items():
                # Convert data to binary blob using pickle
                data_blob = pickle.dumps(data)
                cursor.execute("INSERT OR REPLACE INTO reasoning_cache (query_hash, data, updated_at) VALUES (?, ?, ?)",
                              (query_hash, data_blob, current_time))
            
            # Save CoT results cache
            for query_hash, data in self.cot_results_cache.items():
                # Convert data to binary blob using pickle
                data_blob = pickle.dumps(data)
                cursor.execute("INSERT OR REPLACE INTO cot_results_cache (query_hash, data, updated_at) VALUES (?, ?, ?)",
                              (query_hash, data_blob, current_time))
            
            # Commit transaction
            conn.commit()
            conn.close()
            print(f"[SQLITE] Saved {len(self.reasoning_cache)} reasoning patterns and {len(self.cot_results_cache)} CoT results to database")
            
        except Exception as e:
            print(f"[SQLITE] Error saving CoT data to database: {e}")
    
    def _save_cot_to_files(self):
        """Save CoT-specific caches to pickle files (legacy method)"""
        # Save reasoning cache
        with open(self._get_cache_path(self.cache_config.reasoning_cache_file), 'wb') as f:
            pickle.dump(self.reasoning_cache, f)
        
        # Save CoT results cache
        with open(self._get_cache_path(self.cache_config.cot_results_cache_file), 'wb') as f:
            pickle.dump(self.cot_results_cache, f)

    def _get_reasoning_pattern_hash(self, question_type: str, context_summary: str) -> str:
        """Generate hash for reasoning pattern caching"""
        pattern_str = f"{question_type}_{context_summary}"
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for reasoning pattern reuse"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            return 'definition'
        elif any(word in question_lower for word in ['how', 'process', 'mechanism']):
            return 'process'
        elif any(word in question_lower for word in ['why', 'cause', 'reason']):
            return 'causation'
        elif any(word in question_lower for word in ['treatment', 'cure', 'therapy']):
            return 'treatment'
        elif any(word in question_lower for word in ['symptom', 'sign', 'manifestation']):
            return 'symptoms'
        elif any(word in question_lower for word in ['diagnosis', 'identify', 'detect']):
            return 'diagnosis'
        else:
            return 'general'
    
    def _create_optimized_cot_prompt(self, question: str, context: str, 
                                   question_type: str, step_number: int = 1) -> str:
        """Create optimized CoT prompt based on question type"""
        system_prompt = get_system_prompt()
        
        # Type-specific reasoning templates
        reasoning_templates = {
            'definition': [
                "Understanding: What exactly is being asked to be defined?",
                "Context Analysis: What relevant information is available?",
                "Definition Formation: How can I provide a clear, accurate definition?",
                "Medical Context: What medical context should be included?",
                "Final Answer: Based on the medical context and available information, provide the specific definition requested."
            ],
            'process': [
                "Understanding: What process or mechanism is being asked about?",
                "Step Identification: What are the key steps or stages?",
                "Context Review: What supporting information is available?",
                "Process Explanation: How does this process work?",
                "Final Answer: Explain the specific process or mechanism step by step."
            ],
            'causation': [
                "Understanding: What causal relationship is being explored?",
                "Factor Analysis: What are the potential causes or factors?",
                "Evidence Review: What evidence supports these causes?",
                "Mechanism Explanation: How do these causes lead to the effect?",
                "Final Answer: Clearly explain the causal relationship and underlying mechanisms."
            ],
            'treatment': [
                "Understanding: What treatment information is being requested?",
                "Treatment Options: What treatments are available?",
                "Effectiveness Analysis: How effective are these treatments?",
                "Considerations: What factors should be considered?",
                "Final Answer: Present the specific treatment options and recommendations."
            ],
            'symptoms': [
                "Understanding: What symptoms or signs are being asked about?",
                "Symptom Identification: What are the key symptoms?",
                "Clinical Significance: What do these symptoms indicate?",
                "Differential Considerations: How do these relate to diagnosis?",
                "Final Answer: List and explain the specific symptoms and their clinical significance."
            ],
            'diagnosis': [
                "Understanding: What diagnostic information is being requested?",
                "Diagnostic Criteria: What are the key diagnostic features?",
                "Assessment Methods: How is this condition diagnosed?",
                "Differential Diagnosis: What other conditions should be considered?",
                "Final Answer: Present the diagnostic criteria and assessment methods."
            ],
            'general': [
                "Understanding: What is the core question being asked?",
                "Context Analysis: What relevant information is available?",
                "Knowledge Application: How can medical knowledge be applied?",
                "Synthesis: How can information be combined for an answer?",
                "Final Answer: Based on the analysis above, provide a specific answer to the question."
            ]
        }
        
        steps = reasoning_templates.get(question_type, reasoning_templates['general'])
        
        cot_instructions = f"""
You are an expert medical assistant. Follow these {len(steps)} steps to provide a comprehensive answer:

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}

For each step, explain your thinking process clearly and build upon previous steps.
"""
        
        prompt = f"""{system_prompt}

{cot_instructions}

Context Information:
{context}

Question: {question}

Let's think step by step:

Step {step_number}: {steps[0] if step_number <= len(steps) else 'Continue reasoning'}
"""
        
        return prompt
    
    def _parse_reasoning_steps_optimized(self, llm_response: str, question_type: str) -> List[CoTStep]:
        """Optimized parsing with pattern recognition"""
        steps = []
        lines = llm_response.split('\n')
        current_step = None
        current_reasoning = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Step ') and ':' in line:
                # Save previous step
                if current_step is not None:
                    confidence = self._estimate_step_confidence(current_reasoning, question_type)
                    steps.append(CoTStep(
                        step_number=current_step['number'],
                        description=current_step['description'],
                        reasoning='\n'.join(current_reasoning).strip(),
                        confidence=confidence
                    ))
                
                try:
                    step_parts = line.split(':', 1)
                    step_num = int(step_parts[0].replace('Step', '').strip())
                    step_desc = step_parts[1].strip() if len(step_parts) > 1 else ''
                    current_step = {'number': step_num, 'description': step_desc}
                    current_reasoning = []
                except (ValueError, IndexError):
                    current_reasoning.append(line)
            else:
                if line:  # Skip empty lines
                    current_reasoning.append(line)
        
        if current_step is not None:
            confidence = self._estimate_step_confidence(current_reasoning, question_type)
            steps.append(CoTStep(
                step_number=current_step['number'],
                description=current_step['description'],
                reasoning='\n'.join(current_reasoning).strip(),
                confidence=confidence
            ))
        
        return steps
    
    def _estimate_step_confidence(self, reasoning_lines: List[str], question_type: str) -> float:
        """Estimate confidence based on reasoning quality"""
        if not reasoning_lines:
            return 0.0
        
        reasoning_text = ' '.join(reasoning_lines).lower()
        
        # Quality indicators
        quality_indicators = {
            'medical_terms': ['diagnosis', 'treatment', 'symptom', 'condition', 'patient'],
            'certainty_words': ['clearly', 'definitely', 'established', 'confirmed'],
            'evidence_words': ['research', 'study', 'evidence', 'data', 'clinical'],
            'reasoning_words': ['because', 'therefore', 'thus', 'consequently', 'due to']
        }
        
        score = 0.5  # Base score
        
        for category, words in quality_indicators.items():
            matches = sum(1 for word in words if word in reasoning_text)
            score += min(matches * 0.1, 0.2)  # Cap contribution per category
        
        # Length bonus (longer reasoning often more thorough)
        if len(reasoning_text) > 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_overall_confidence(self, steps: List[CoTStep]) -> float:
        """Calculate overall confidence from individual steps"""
        if not steps:
            return 0.0
        
        step_confidences = [step.confidence or 0.5 for step in steps]
        
        # Weighted average with higher weight for later steps
        weights = [i + 1 for i in range(len(step_confidences))]
        weighted_sum = sum(conf * weight for conf, weight in zip(step_confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def query(self, question: str, param: QueryParam = QueryParam()) -> CoTResult:
        """Optimized CoT query with caching and parallel processing"""
        # Check CoT results cache first
        cot_query_hash = self._get_query_hash(f"cot_{question}", param)
        if cot_query_hash in self.cot_results_cache:
            print(f"[COT_CACHE] CoT result cache hit")
            return self.cot_results_cache[cot_query_hash]
        
        try:
            # Step 1: Get embeddings and retrieve context (parallel)
            loop = asyncio.get_event_loop()
            q_emb_task = loop.run_in_executor(
                self.executor, 
                self.embedding_func, 
                [question]
            )
            
            # Classify question type for optimized reasoning
            question_type = self._classify_question_type(question)
            
            # Wait for embedding
            q_emb_arr = await q_emb_task
            
            if isinstance(q_emb_arr, np.ndarray):
                q_emb = q_emb_arr[0]
            elif isinstance(q_emb_arr, list) and isinstance(q_emb_arr[0], (list, np.ndarray)):
                q_emb = np.array(q_emb_arr[0])
            else:
                q_emb = np.array(q_emb_arr)
            
            # Step 2: Retrieve relevant context
            top_k = param.top_k
            top_idx = self.vector_search(q_emb, n_results=top_k)
            
            if len(top_idx) == 0:
                return CoTResult(
                    final_answer="No relevant information found in knowledge base.",
                    reasoning_steps=[],
                    total_steps=0,
                    confidence_score=0.0,
                    sources_used=[],
                    reasoning_chain="No context available for reasoning."
                )
            
            # Get context
            context_chunks = [self.chunks[i] for i in top_idx if i < len(self.chunks)]
            context = "\n\n".join(context_chunks)
            context_summary = context[:200] + "..." if len(context) > 200 else context
            
            # Step 3: Check for cached reasoning patterns
            pattern_hash = self._get_reasoning_pattern_hash(question_type, context_summary)
            cached_reasoning_template = self.reasoning_cache.get(pattern_hash)
            
            # Step 4: Generate reasoning steps
            reasoning_steps = []
            sources_used = top_idx
            
            # Create optimized prompt
            prompt = self._create_optimized_cot_prompt(question, context, question_type, 1)
            
            # Generate reasoning with LLM
            llm_response = await loop.run_in_executor(
                self.executor,
                self.llm_model_func,
                prompt
            )
            
            # Parse reasoning steps
            reasoning_steps = self._parse_reasoning_steps_optimized(llm_response, question_type)
            # Calculate confidence
            overall_confidence = self._calculate_overall_confidence(reasoning_steps)
            # Step 5: Extract final answer
            final_answer = self._extract_final_answer(llm_response, reasoning_steps)
            # Step 6: Extract patient-friendly answer (if available)
            patient_friendly_answer = self._extract_patient_friendly_answer(llm_response)
            # Step 7: Create reasoning chain
            reasoning_chain = self._create_reasoning_chain(reasoning_steps)
            # Create result
            result = CoTResult(
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                total_steps=len(reasoning_steps),
                confidence_score=overall_confidence,
                sources_used=sources_used,
                reasoning_chain=reasoning_chain
            )
            result.patient_friendly_answer = patient_friendly_answer
            
            # Cache the result
            self.cot_results_cache[cot_query_hash] = result
            
            # Cache reasoning pattern if high confidence
            if overall_confidence > 0.8 and self.cache_config.enable_reasoning_cache:
                self.reasoning_cache[pattern_hash] = {
                    'question_type': question_type,
                    'reasoning_template': [step.description for step in reasoning_steps],
                    'confidence': overall_confidence
                }
            
            # Save caches periodically
            if len(self.cot_results_cache) % 10 == 0:
                self._save_cot_cache()
            
            # Limit cache sizes
            if len(self.cot_results_cache) > 500:
                oldest_keys = list(self.cot_results_cache.keys())[:50]
                for key in oldest_keys:
                    del self.cot_results_cache[key]
            
            if len(self.reasoning_cache) > 100:
                oldest_keys = list(self.reasoning_cache.keys())[:10]
                for key in oldest_keys:
                    del self.reasoning_cache[key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in CoT query: {e}")
            return CoTResult(
                final_answer=f"Error during reasoning: {str(e)}",
                reasoning_steps=[],
                total_steps=0,
                confidence_score=0.0,
                sources_used=[],
                reasoning_chain=f"Error: {str(e)}"
            )
    
    def _extract_final_answer(self, llm_response: str, reasoning_steps: List[CoTStep]) -> str:
        """Extract final answer from LLM response with improved handling"""
        if not llm_response:
            return "No response generated"
        
        # Template instructions to remove (more comprehensive list)
        template_instructions = [
            "Provide a comprehensive definition.",
            "Provide a clear process explanation.", 
            "Explain the causal relationship.",
            "Provide treatment recommendations.",
            "Describe the symptoms and their significance.",
            "Provide diagnostic guidance.",
            "Provide a comprehensive response.",
            "Based on the medical context and available information, provide the specific definition requested.",
            "Explain the specific process or mechanism step by step.",
            "Clearly explain the causal relationship and underlying mechanisms.",
            "Present the specific treatment options and recommendations.",
            "List and explain the specific symptoms and their clinical significance.",
            "Present the diagnostic criteria and assessment methods.",
            "Based on the analysis above, provide a specific answer to the question.",
            "Let's think step by step:",
            "Step 1:", "Step 2:", "Step 3:", "Step 4:", "Step 5:"
        ]
        
        lines = llm_response.split('\n')
        
        for line in lines:
            if 'final answer' in line.lower() and ':' in line:
                # Remove label and return only the answer content
                answer = line.split(':', 1)[1].strip()
                
                # Remove template instructions
                for instruction in template_instructions:
                    if instruction.lower() in answer.lower():
                        answer = answer.replace(instruction, "").strip()
                
                # If answer is empty or just template text, look for content in subsequent lines
                if not answer or answer.lower() in ["final answer:", "answer:"] or len(answer) < 10:
                    next_line_index = lines.index(line) + 1
                    extracted_lines = []
                    
                    for next_line in lines[next_line_index:]:
                        clean_line = next_line.strip()
                        if clean_line:
                            # Skip template instructions and step markers
                            if (not any(instr.lower() in clean_line.lower() for instr in template_instructions) and 
                                not clean_line.lower().startswith(('step ', 'understanding:', 'context analysis:', 'definition formation:', 
                                                                   'medical context:', 'process explanation:', 'factor analysis:', 
                                                                   'evidence review:', 'mechanism explanation:'))):
                                extracted_lines.append(clean_line)
                    
                    if extracted_lines:
                        answer = ' '.join(extracted_lines)
                
                # Check if the answer is just a template instruction and not actual content
                if answer in template_instructions:
                    continue
                
                # Check if the answer starts with any template instruction
                if any(answer.startswith(instr) for instr in template_instructions):
                    for instr in template_instructions:
                        if answer.startswith(instr):
                            answer = answer[len(instr):].strip()
                            break
                
                if answer and len(answer.strip()) > 10:
                    return answer.strip()
        
        # Fallback: look for substantial content after reasoning steps
        content_lines = []
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue
                
            # Skip template instructions and step markers
            if (any(instr.lower() in clean_line.lower() for instr in template_instructions) or
                clean_line.lower().startswith(('step ', 'understanding:', 'context analysis:', 'definition formation:', 
                                               'medical context:', 'process explanation:', 'factor analysis:', 
                                               'evidence review:', 'mechanism explanation:', 'let\'s think'))):
                continue
                
            # Look for substantial medical content
            if len(clean_line) > 20 and any(word in clean_line.lower() for word in 
                                           ['cancer', 'skin', 'melanoma', 'treatment', 'diagnosis', 'symptom', 'medical', 'patient', 'disease', 'condition']):
                content_lines.append(clean_line)
        
        if content_lines:
            return ' '.join(content_lines[:3])  # Take first 3 substantial lines
        
        # If no explicit final answer, use last reasoning step
        if reasoning_steps:
            return reasoning_steps[-1].reasoning
        
        # Last resort: return last substantial paragraph
        paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip() and len(p.strip()) > 20]
        if paragraphs:
            last_paragraph = paragraphs[-1]
            # Remove template instructions from last paragraph
            for instruction in template_instructions:
                if instruction.lower() in last_paragraph.lower():
                    last_paragraph = last_paragraph.replace(instruction, "").strip()
            
            if last_paragraph and len(last_paragraph) > 20:
                return last_paragraph
        
        return "Unable to generate final answer."

    def _extract_patient_friendly_answer(self, llm_response: str) -> str:
        """Extract patient-friendly answer from LLM response"""
        lines = llm_response.split('\n')
        for line in lines:
            if 'patient-friendly answer' in line.lower() and ':' in line:
                return line.split(':', 1)[1].strip()
        return ""
    
    def _create_reasoning_chain(self, reasoning_steps: List[CoTStep]) -> str:
        """Create a readable reasoning chain"""
        if not reasoning_steps:
            return "No reasoning steps available."
        
        chain_parts = []
        for step in reasoning_steps:
            chain_parts.append(f"Step {step.step_number}: {step.description}")
            if step.reasoning:
                chain_parts.append(f"Reasoning: {step.reasoning[:200]}..." if len(step.reasoning) > 200 else f"Reasoning: {step.reasoning}")
            chain_parts.append("")  # Empty line for separation
        
        return "\n".join(chain_parts)
    
    async def query_simple(self, question: str, param: QueryParam = QueryParam()) -> str:
        """Simplified query method for backward compatibility"""
        result = await self.query(question, param)
        return result.final_answer
    
    def get_cot_stats(self) -> Dict:
        """Get CoT-specific performance statistics"""
        base_stats = self.get_stats()
        cot_stats = {
            'reasoning_cache_size': len(self.reasoning_cache),
            'cot_results_cache_size': len(self.cot_results_cache),
            'reasoning_cache_enabled': self.cache_config.enable_reasoning_cache,
            'max_reasoning_steps': self.max_reasoning_steps,
            'confidence_threshold': self.confidence_threshold
        }
        return {**base_stats, **cot_stats}
    
    def clear_cot_cache(self):
        """Clear CoT-specific caches"""
        # Clear CoT-specific caches
        self.reasoning_cache = {}
        self.cot_results_cache = {}
        
        if self.cache_config.use_sqlite:
            try:
                conn = sqlite3.connect(self._get_db_path())
                cursor = conn.cursor()
                
                # Clear CoT tables
                cursor.execute("DELETE FROM reasoning_cache")
                cursor.execute("DELETE FROM cot_results_cache")
                
                conn.commit()
                conn.close()
                print("[SQLITE] Reset CoT database cache")
            except Exception as e:
                print(f"[SQLITE] Error resetting CoT database: {e}")
        else:
            # Remove CoT cache files
            cot_cache_files = [
                self.cache_config.reasoning_cache_file,
                self.cache_config.cot_results_cache_file
            ]
            
            for cache_file in cot_cache_files:
                cache_path = self._get_cache_path(cache_file)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"[CACHE] Removed {cache_path}")
        
        print("[CACHE] All CoT caches cleared")

def create_optimized_cot_rag(working_dir: str = "./cot_rag_cache", 
                         llm_model_func=None, 
                         embedding_func=None,
                         max_reasoning_steps: int = 5,
                         confidence_threshold: float = 0.7,
                         enable_cache: bool = True,
                         use_sqlite: bool = True) -> OptimizedCoTRAG:
    """Create an optimized CoT-RAG instance with default settings"""
    if embedding_func is None:
        from rag.llm import hugging_face_embedding
        embedding_func = hugging_face_embedding
    if llm_model_func is None:
        llm_model_func = lambda prompt: prompt  # Default passthrough
    
    cache_config = CoTCacheConfig(
        cache_dir=working_dir,
        enable_cache=enable_cache,
        use_sqlite=use_sqlite
    )
    
    return OptimizedCoTRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        max_reasoning_steps=max_reasoning_steps,
        confidence_threshold=confidence_threshold,
        cache_config=cache_config
    )
