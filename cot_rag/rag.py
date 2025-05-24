import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rag.rag import RAG, QueryParam
from rag.prompt import get_system_prompt

@dataclass
class CoTStep:
    """Represents a single step in the chain of thought reasoning."""
    step_number: int
    description: str
    reasoning: str
    retrieved_context: Optional[str] = None
    confidence: Optional[float] = None

@dataclass
class CoTResult:
    """Enhanced result structure for Chain of Thought RAG."""
    final_answer: str
    reasoning_steps: List[CoTStep]
    total_steps: int
    confidence_score: float
    sources_used: List[int]
    reasoning_chain: str

class CoTRAG(RAG):
    """
    Enhanced Chain-of-Thought RAG implementation following modern CoT standards.
    
    Features:
    - Step-by-step reasoning decomposition
    - Explicit reasoning chain generation
    - Multi-stage retrieval and refinement
    - Enhanced error handling and validation
    - Confidence scoring for reasoning steps
    """
    
    def __init__(self, working_dir: str, llm_model_func, embedding_func, 
                 max_reasoning_steps: int = 5, confidence_threshold: float = 0.7):
        super().__init__(working_dir, llm_model_func, embedding_func)
        self.max_reasoning_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def _create_cot_prompt(self, question: str, context: str, step_number: int = 1) -> str:
        """Create a structured Chain of Thought prompt following best practices."""
        system_prompt = get_system_prompt()
        
        cot_instructions = """
You are an expert medical assistant. Follow these steps to provide a comprehensive answer:

1. Understanding: First, clearly understand what the question is asking
2. Analysis: Break down the question into key components
3. Context Review: Examine the provided context for relevant information
4. Reasoning: Think through the problem step by step
5. Synthesis: Combine your reasoning with the context to form a conclusion
6. Final Answer: Provide a clear, concise final answer

For each step, explain your thinking process clearly."""
        
        prompt = f"""{system_prompt}

{cot_instructions}

Context Information:
{context}

Question: {question}

Let's think step by step:

Step {step_number}: Understanding the Question
"""
        
        return prompt
    
    def _parse_reasoning_steps(self, llm_response: str) -> List[CoTStep]:
        """Parse the LLM response to extract individual reasoning steps."""
        steps = []
        lines = llm_response.split('\n')
        current_step = None
        current_reasoning = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Step ') and ':' in line:
                # Save previous step if exists
                if current_step is not None:
                    steps.append(CoTStep(
                        step_number=current_step['number'],
                        description=current_step['description'],
                        reasoning='\n'.join(current_reasoning).strip()
                    ))
                
                # Start new step
                try:
                    step_parts = line.split(':', 1)
                    step_num = int(step_parts[0].replace('Step', '').strip())
                    step_desc = step_parts[1].strip()
                    current_step = {'number': step_num, 'description': step_desc}
                    current_reasoning = []
                except (ValueError, IndexError):
                    current_reasoning.append(line)
            else:
                if line:
                    current_reasoning.append(line)
        
        # Add final step
        if current_step is not None:
            steps.append(CoTStep(
                step_number=current_step['number'],
                description=current_step['description'],
                reasoning='\n'.join(current_reasoning).strip()
            ))
        
        return steps
    
    def _extract_final_answer(self, llm_response: str) -> str:
        """Extract the final answer from the LLM response."""
        # Look for common final answer patterns
        patterns = [
            'Final Answer:',
            'Conclusion:',
            'Answer:',
            'Therefore:',
            'In summary:'
        ]
        
        lines = llm_response.split('\n')
        final_answer_lines = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            if any(pattern in line for pattern in patterns):
                capturing = True
                # Include the line after the pattern
                answer_part = line.split(':', 1)
                if len(answer_part) > 1 and answer_part[1].strip():
                    final_answer_lines.append(answer_part[1].strip())
            elif capturing and line:
                final_answer_lines.append(line)
            elif capturing and not line:
                break  # Stop at empty line after final answer
        
        if final_answer_lines:
            return ' '.join(final_answer_lines)
        
        # Fallback: return last non-empty paragraph
        paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip()]
        return paragraphs[-1] if paragraphs else llm_response.strip()
    
    def _calculate_confidence(self, reasoning_steps: List[CoTStep], 
                            context_relevance: float) -> float:
        """Calculate confidence score based on reasoning quality and context relevance."""
        if not reasoning_steps:
            return 0.0
        
        # Base confidence from number of reasoning steps
        step_confidence = min(len(reasoning_steps) / self.max_reasoning_steps, 1.0)
        
        # Average reasoning length (longer reasoning often indicates more thorough thinking)
        avg_reasoning_length = sum(len(step.reasoning) for step in reasoning_steps) / len(reasoning_steps)
        length_confidence = min(avg_reasoning_length / 100, 1.0)  # Normalize to 100 chars
        
        # Combine factors
        overall_confidence = (step_confidence * 0.4 + 
                            length_confidence * 0.3 + 
                            context_relevance * 0.3)
        
        return min(overall_confidence, 1.0)

    async def query(self, question: str, param: QueryParam = QueryParam()) -> CoTResult:
        """
        Enhanced Chain-of-Thought query with structured reasoning and result validation.
        
        Args:
            question: The input question
            param: Query parameters including top_k for retrieval
            
        Returns:
            CoTResult: Structured result with reasoning steps and final answer
        """
        try:
            # Step 1: Embed question
            q_emb_arr = self.embedding_func([question])
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
                    final_answer="No relevant information found in the knowledge base.",
                    reasoning_steps=[],
                    total_steps=0,
                    confidence_score=0.0,
                    sources_used=[],
                    reasoning_chain="No context available for reasoning."
                )

            # Step 3: Build context and calculate relevance
            context_chunks = [self.chunks[i] for i in top_idx]
            context = "\n\n".join(context_chunks)
            
            # Simple context relevance calculation (can be enhanced)
            question_words = set(question.lower().split())
            context_words = set(context.lower().split())
            context_relevance = len(question_words.intersection(context_words)) / len(question_words) if question_words else 0.0

            # Step 4: Generate Chain-of-Thought reasoning
            cot_prompt = self._create_cot_prompt(question, context)
            
            if asyncio.iscoroutinefunction(self.llm_model_func):
                llm_response = await self.llm_model_func(cot_prompt)
            else:
                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(None, self.llm_model_func, cot_prompt)
            
            # Step 5: Parse reasoning steps
            reasoning_steps = self._parse_reasoning_steps(llm_response)
            
            # Step 6: Extract final answer
            final_answer = self._extract_final_answer(llm_response)
            
            # Step 7: Calculate confidence
            confidence_score = self._calculate_confidence(reasoning_steps, context_relevance)
            
            # Step 8: Create structured result
            result = CoTResult(
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                total_steps=len(reasoning_steps),
                confidence_score=confidence_score,
                sources_used=top_idx.tolist() if hasattr(top_idx, 'tolist') else list(top_idx),
                reasoning_chain=llm_response
            )
            
            self.logger.info(f"CoT query completed with {len(reasoning_steps)} steps, confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in CoT query: {str(e)}")
            return CoTResult(
                final_answer=f"An error occurred while processing your question: {str(e)}",
                reasoning_steps=[],
                total_steps=0,
                confidence_score=0.0,
                sources_used=[],
                reasoning_chain=f"Error: {str(e)}"
            )
    
    async def query_simple(self, question: str, param: QueryParam = QueryParam()) -> str:
        """Simplified query method that returns just the final answer for backward compatibility."""
        result = await self.query(question, param)
        return result.final_answer

def load_and_insert_data_cotrag(cotrag, data_dir):
    """
    Hàm này dùng để load dữ liệu và insert vào CoTRAG instance.
    """
    # Giả định có hàm load_data_from_dir trả về list các chunk
    from rag.utils import load_data_from_dir
    chunks = load_data_from_dir(data_dir)
    cotrag.insert_chunks(chunks)

# Định nghĩa hàm cotrag_query
async def cotrag_query(query, cotrag):
    param = QueryParam(top_k=5)
    return await cotrag.query(query, param)