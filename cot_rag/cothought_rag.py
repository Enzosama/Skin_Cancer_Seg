import argparse
import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv()
import torch
from cot_rag import CoTRAG, CoTResult
from rag import QueryParam
from rag.llm import get_llm_func, get_embedding_func
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_and_insert_data_cotrag(co_trag, data_path):
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for fname in files:
                path = os.path.join(root, fname)
                ext = fname.lower().rsplit('.', 1)[-1]
                if ext == 'pdf':
                    try:
                        reader = PdfReader(path)
                        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
                    except Exception as e:
                        print(f"Failed to read PDF {path}: {e}", file=sys.stderr)
                        continue
                elif ext in ('txt', 'csv'):
                    with open(path, encoding='utf-8') as f:
                        text = f.read()
                else:
                    continue
                co_trag.insert(text)
    else:
        with open(data_path, encoding='utf-8') as f:
            text = f.read()
        co_trag.insert(text)

async def cotrag_query(question, co_trag, top_k=5, detailed=False):
    """Enhanced CoT query function with support for detailed reasoning output.
    
    Args:
        question: The input question
        co_trag: CoTRAG instance
        top_k: Number of top results to retrieve
        detailed: If True, returns detailed reasoning steps; if False, returns simple answer
    
    Returns:
        String response (simple or detailed based on detailed parameter)
    """
    param = QueryParam(top_k=top_k)
    
    try:
        # Use the enhanced query method
        result = await co_trag.query(question, param)
        
        if isinstance(result, CoTResult):
            if detailed:
                # Format detailed response with reasoning steps
                detailed_response = []
                detailed_response.append(f"Question: {question}")
                detailed_response.append(f"Confidence Score: {result.confidence_score:.2f}")
                detailed_response.append(f"Sources Used: {len(result.sources_used)} documents")
                detailed_response.append("\n=== REASONING PROCESS ===")
                
                for i, step in enumerate(result.reasoning_steps, 1):
                    detailed_response.append(f"\nStep {step.step_number}: {step.description}")
                    detailed_response.append(f"Reasoning: {step.reasoning}")
                
                detailed_response.append("\n=== FINAL ANSWER ===")
                detailed_response.append(result.final_answer)
                
                return "\n".join(detailed_response)
            else:
                # Return simple answer for backward compatibility
                return result.final_answer
        else:
            # Fallback for legacy string responses
            if isinstance(result, (list, tuple)):
                if all(isinstance(x, (float, int)) for x in result):
                    return "Unable to return vector result, please try with a different question."
                return '\n'.join(str(x) for x in result)
            return str(result) if result else "No suitable result found"
            
    except Exception as e:
        logging.error(f"Error in cotrag_query: {str(e)}")
        return f"An error occurred while processing your question: {str(e)}"

# Backward compatibility function
async def cotrag_query_simple(question, co_trag, top_k=5):
    """Simple query function for backward compatibility."""
    return await cotrag_query(question, co_trag, top_k, detailed=False)

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced CoT-RAG on a Skin Cancer dataset.")
    parser.add_argument("--working_dir", default="./rag_cache", help="Working directory for cache")
    parser.add_argument("--data_file", default="Data", help="Path to data file or directory")
    parser.add_argument("--question", required=True, help="Question to ask the CoT-RAG system")
    parser.add_argument("--engine", choices=["google", "llama", "hugging_face", "openai_github"], 
                       default="google", help="LLM engine to use")
    parser.add_argument("--embed_engine", choices=["openai", "google", "groq", "hugging_face", "openai_github"], 
                       default="openai", help="Embedding engine to use")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed reasoning steps in output")
    parser.add_argument("--max_reasoning_steps", type=int, default=5, 
                       help="Maximum number of reasoning steps")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, 
                       help="Confidence threshold for reasoning")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        llm_func = get_llm_func(args.engine)
        emb_func = get_embedding_func(args.embed_engine)

        # Create enhanced CoTRAG instance with new parameters
        co_trag = CoTRAG(
            working_dir=args.working_dir,
            llm_model_func=llm_func,
            embedding_func=emb_func,
            max_reasoning_steps=args.max_reasoning_steps,
            confidence_threshold=args.confidence_threshold
        )

        # Load data
        data_path = args.data_file
        if args.debug:
            print(f"[DEBUG] Loading data from: {data_path}")
        
        load_and_insert_data_cotrag(co_trag, data_path)
        
        if args.debug:
            print(f"[DEBUG] Loaded {len(co_trag.chunks)} chunks into CoTRAG")

        import asyncio
        async def enhanced_query():
            if args.debug:
                print(f"Processing question: {args.question}")
                print(f"Using detailed output: {args.detailed}")
            
            result = await cotrag_query(
                args.question, 
                co_trag, 
                top_k=args.top_k, 
                detailed=args.detailed
            )
            
            return result

        result = asyncio.run(enhanced_query())
        
        # Output results
        if args.detailed:
            print("\n" + "="*60)
            print("ENHANCED CHAIN-OF-THOUGHT RAG RESULT")
            print("="*60)
            print(result)
            print("="*60)
        else:
            print("\n[CoT-RAG Answer]:", result)
            
        if args.debug:
            print(f"\n[DEBUG] Query completed successfully")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
