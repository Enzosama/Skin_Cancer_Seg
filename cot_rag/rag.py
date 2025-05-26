import numpy as np
import asyncio
import logging
import re
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
You are an expert dermatologist specializing in skin cancer. Follow these steps to provide a comprehensive answer:

1. Understanding: First, clearly understand what the question is asking about skin conditions or cancer
2. Analysis: Break down the question into key medical components and identify relevant skin cancer concepts
3. Context Review: Carefully examine the provided medical context for relevant diagnostic or treatment information
4. Medical Reasoning: Think through the medical aspects step by step, considering differential diagnoses when appropriate
5. Evidence-Based Synthesis: Combine your reasoning with the medical literature to form a conclusion supported by evidence
6. Patient-Friendly Answer: Provide a clear, concise final answer that would be appropriate for patient education

For each step, explain your medical reasoning process clearly, citing specific information from the provided context.

When analyzing skin conditions:
- Consider the ABCDE criteria for melanoma evaluation when relevant
- Distinguish between benign and malignant characteristics
- Reference appropriate diagnostic procedures and treatment options
- Note when further medical consultation would be necessary
"""
        
        # Check if context contains medical document format and enhance the prompt
        if "Title:" in context and any(term in context.lower() for term in ["melanoma", "carcinoma", "nevus", "mole", "skin cancer", "lesion"]):
            medical_context_note = """
Note: The provided context contains medical information from reputable sources. Pay special attention to:
- Clinical descriptions of skin conditions
- Diagnostic criteria and warning signs
- Treatment recommendations and guidelines
- Statistical information about prevalence and risk factors
"""
            cot_instructions += medical_context_note
        
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
        """Parse the LLM response to extract individual reasoning steps with enhanced medical content handling."""
        steps = []
        lines = llm_response.split('\n')
        current_step = None
        current_reasoning = []
        current_context = []
        in_context_section = False
        confidence_score = None
        
        for line in lines:
            line = line.strip()
            
            # Check for confidence indicators in medical reasoning
            if any(term in line.lower() for term in ['confidence:', 'certainty:', 'confidence level:', 'medical certainty:']):
                try:
                    # Extract confidence value if present (e.g., "Confidence: 0.8" or "Confidence: High (0.9)")
                    conf_match = re.search(r'\b(\d+(\.\d+)?)\b', line)
                    if conf_match:
                        confidence_score = float(conf_match.group(1))
                        if confidence_score > 1.0:  # Normalize if on a 0-100 scale
                            confidence_score /= 100.0
                except (ValueError, AttributeError):
                    pass
            
            # Detect context reference sections
            if any(marker in line.lower() for marker in ['from context:', 'reference:', 'according to:', 'source states:', 'medical literature:']):
                in_context_section = True
                continue
            elif in_context_section and (not line or line.startswith('Step ') or 'conclusion:' in line.lower()):
                in_context_section = False
            
            if line.startswith('Step ') and ':' in line:
                # Save previous step if exists
                if current_step is not None:
                    steps.append(CoTStep(
                        step_number=current_step['number'],
                        description=current_step['description'],
                        reasoning='\n'.join(current_reasoning).strip(),
                        retrieved_context='\n'.join(current_context).strip() if current_context else None,
                        confidence=confidence_score
                    ))
                
                # Reset for new step
                current_context = []
                confidence_score = None
                in_context_section = False
                
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
                    if in_context_section:
                        current_context.append(line)
                    else:
                        current_reasoning.append(line)
        
        # Add final step
        if current_step is not None:
            steps.append(CoTStep(
                step_number=current_step['number'],
                description=current_step['description'],
                reasoning='\n'.join(current_reasoning).strip(),
                retrieved_context='\n'.join(current_context).strip() if current_context else None,
                confidence=confidence_score
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
        """Enhanced query method with Chain of Thought reasoning optimized for medical content."""
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
                    final_answer="No relevant medical information found in the knowledge base.",
                    reasoning_steps=[],
                    total_steps=0,
                    confidence_score=0.0,
                    sources_used=[],
                    reasoning_chain="No context available for reasoning."
                )

            # Step 3: Build context and calculate relevance
            context_chunks = [self.chunks[i] for i in top_idx]
            
            # For medical content, enhance context with source information
            enhanced_chunks = []
            for i, chunk in enumerate(context_chunks):
                # Check if this is from the medical content format
                if chunk.startswith("Title:"):
                    # Add source number for reference
                    enhanced_chunks.append(f"Source {i+1}:\n{chunk}")
                else:
                    enhanced_chunks.append(chunk)
            
            context = "\n\n".join(enhanced_chunks)
            
            # Detect if this is a medical question
            is_medical_question = any(term in question.lower() for term in [
                "skin", "cancer", "melanoma", "carcinoma", "mole", "nevus", "lesion", 
                "dermatology", "biopsy", "treatment", "diagnosis", "symptom", "abcde"
            ])
            
            # Simple context relevance calculation
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
            
            # Step 7: Calculate confidence with enhanced medical content awareness
            confidence_score = self._calculate_confidence(reasoning_steps, context_relevance)
            
            # For medical content, adjust confidence based on source quality
            if is_medical_question:
                medical_source_bonus = 0.0
                for chunk in context_chunks:
                    if any(term in chunk.lower() for term in ["yale medicine", "mayo clinic", "nih", "pubmed", "journal", "research"]):
                        medical_source_bonus += 0.05  # Bonus for reputable medical sources
                
                confidence_score = min(1.0, confidence_score + medical_source_bonus)  # Cap at 1.0
                
                # For medical content, add a disclaimer if confidence is low
                if confidence_score < 0.7:
                    final_answer += "\n\nNote: This information is provided for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for diagnosis and treatment."
            
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