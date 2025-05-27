#!/usr/bin/env python3
"""
RAG Performance Benchmark Script
Compares optimized RAG vs original RAG performance
"""

import time
import asyncio
import statistics
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import both original and optimized versions
from rag.rag import RAG, QueryParam
from cot_rag.rag import CoTRAG
from optimized_rag import OptimizedRAG, create_optimized_rag
from optimized_cot_rag import OptimizedCoTRAG, create_optimized_cot_rag
from rag.llm import hugging_face_embedding, get_llm_func
from native_rag import load_and_insert_data

@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    test_name: str
    query: str
    response_time: float
    memory_usage: float
    cache_hits: int
    cache_misses: int
    result_length: int
    success: bool
    error_message: str = ""

class RAGBenchmark:
    """Benchmark suite for RAG performance testing"""
    
    def __init__(self, data_dir: str = "./Crawl_data"):
        self.data_dir = data_dir
        self.test_queries = [
            "What is melanoma?",
            "What causes skin cancer?",
            "How is melanoma diagnosed?",
            "What are the symptoms of skin cancer?",
            "What are the treatment options for melanoma?",
            "How can I prevent skin cancer?",
            "What is the difference between melanoma and other skin cancers?",
            "What are the risk factors for skin cancer?",
            "How effective is immunotherapy for melanoma?",
            "What is the prognosis for early-stage melanoma?"
        ]
        
        # Initialize LLM function
        self.llm_func = get_llm_func("hugging_face")
        self.embedding_func = hugging_face_embedding
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0 
    
    def benchmark_original_rag(self, queries: List[str]) -> List[BenchmarkResult]:
        """Benchmark original RAG implementation"""
        print("\n=== Benchmarking Original RAG ===")
        results = []
        
        for i, query in enumerate(queries):
            print(f"\nTest {i+1}/{len(queries)}: {query}")
            
            try:
                memory_before = self.measure_memory_usage()
                start_time = time.time()
                
                rag = RAG(
                    working_dir="./rag_cache_original",
                    llm_model_func=self.llm_func,
                    embedding_func=self.embedding_func
                )
                
                # Load data
                load_and_insert_data(rag, self.data_dir)
                
                # Query
                param = QueryParam(top_k=5)
                result = rag.query(query, param)
                
                end_time = time.time()
                
                # Measure memory after
                memory_after = self.measure_memory_usage()
                
                response_time = end_time - start_time
                memory_usage = memory_after - memory_before
                
                benchmark_result = BenchmarkResult(
                    test_name="Original RAG",
                    query=query,
                    response_time=response_time,
                    memory_usage=memory_usage,
                    cache_hits=0,  # Original doesn't have cache stats
                    cache_misses=0,
                    result_length=len(result) if result else 0,
                    success=bool(result and len(result.strip()) > 10)
                )
                
                results.append(benchmark_result)
                print(f"  Time: {response_time:.2f}s, Memory: {memory_usage:.1f}MB, Success: {benchmark_result.success}")
                
            except Exception as e:
                error_result = BenchmarkResult(
                    test_name="Original RAG",
                    query=query,
                    response_time=0.0,
                    memory_usage=0.0,
                    cache_hits=0,
                    cache_misses=0,
                    result_length=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                print(f"  Error: {e}")
        
        return results
    
    def benchmark_optimized_rag(self, queries: List[str]) -> List[BenchmarkResult]:
        """Benchmark optimized RAG implementation"""
        print("\n=== Benchmarking Optimized RAG ===")
        results = []
        
        for i, query in enumerate(queries):
            print(f"\nTest {i+1}/{len(queries)}: {query}")
            
            try:
                # Measure memory before
                memory_before = self.measure_memory_usage()
                
                # Time the operation
                start_time = time.time()
                
                # Create optimized RAG instance
                rag = create_optimized_rag(
                    working_dir="./rag_cache_optimized",
                    llm_model_func=self.llm_func,
                    embedding_func=self.embedding_func,
                    enable_cache=True
                )
                
                # Load data (with caching)
                load_and_insert_data(rag, self.data_dir)
                
                # Query
                param = QueryParam(top_k=5)
                result = rag.query(query, param)
                
                end_time = time.time()
                
                # Measure memory after
                memory_after = self.measure_memory_usage()
                
                response_time = end_time - start_time
                memory_usage = memory_after - memory_before
                
                # Get cache stats
                stats = rag.get_stats()
                
                benchmark_result = BenchmarkResult(
                    test_name="Optimized RAG",
                    query=query,
                    response_time=response_time,
                    memory_usage=memory_usage,
                    cache_hits=stats.get('query_cache_hits', 0),
                    cache_misses=stats.get('query_cache_misses', 0),
                    result_length=len(result) if result else 0,
                    success=bool(result and len(result.strip()) > 10)
                )
                
                results.append(benchmark_result)
                print(f"  Time: {response_time:.2f}s, Memory: {memory_usage:.1f}MB, Cache: {benchmark_result.cache_hits}H/{benchmark_result.cache_misses}M, Success: {benchmark_result.success}")
                
            except Exception as e:
                error_result = BenchmarkResult(
                    test_name="Optimized RAG",
                    query=query,
                    response_time=0.0,
                    memory_usage=0.0,
                    cache_hits=0,
                    cache_misses=0,
                    result_length=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                print(f"  Error: {e}")
        
        return results
    
    async def benchmark_original_cot_rag(self, queries: List[str]) -> List[BenchmarkResult]:
        """Benchmark original CoT-RAG implementation"""
        print("\n=== Benchmarking Original CoT-RAG ===")
        results = []
        
        for i, query in enumerate(queries[:5]):  # Limit to 5 queries for CoT (slower)
            print(f"\nTest {i+1}/5: {query}")
            
            try:
                # Measure memory before
                memory_before = self.measure_memory_usage()
                
                # Time the operation
                start_time = time.time()
                
                # Create CoT-RAG instance
                cot_rag = CoTRAG(
                    working_dir="./rag_cache_cot_original",
                    llm_model_func=self.llm_func,
                    embedding_func=self.embedding_func
                )
                
                # Load data
                load_and_insert_data(cot_rag, self.data_dir)
                
                # Query
                param = QueryParam(top_k=5)
                result = await cot_rag.query(query, param)
                
                end_time = time.time()
                
                # Measure memory after
                memory_after = self.measure_memory_usage()
                
                response_time = end_time - start_time
                memory_usage = memory_after - memory_before
                
                final_answer = ""
                if result and hasattr(result, 'final_answer'):
                    final_answer = result.final_answer
                elif result and hasattr(result, 'answer'):
                    final_answer = result.answer
                
                benchmark_result = BenchmarkResult(
                    test_name="Original CoT-RAG",
                    query=query,
                    response_time=response_time,
                    memory_usage=memory_usage,
                    cache_hits=0,
                    cache_misses=0,
                    result_length=len(final_answer) if final_answer else 0,
                    success=bool(final_answer and len(final_answer.strip()) > 10)
                )
                
                results.append(benchmark_result)
                print(f"  Time: {response_time:.2f}s, Memory: {memory_usage:.1f}MB, Success: {benchmark_result.success}")
                
            except Exception as e:
                error_result = BenchmarkResult(
                    test_name="Original CoT-RAG",
                    query=query,
                    response_time=0.0,
                    memory_usage=0.0,
                    cache_hits=0,
                    cache_misses=0,
                    result_length=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                print(f"  Error: {e}")
        
        return results
    
    async def benchmark_optimized_cot_rag(self, queries: List[str]) -> List[BenchmarkResult]:
        """Benchmark optimized CoT-RAG implementation"""
        print("\n=== Benchmarking Optimized CoT-RAG ===")
        results = []
        
        for i, query in enumerate(queries[:5]):  # Limit to 5 queries for CoT (slower)
            print(f"\nTest {i+1}/5: {query}")
            
            try:
                # Measure memory before
                memory_before = self.measure_memory_usage()
                
                # Time the operation
                start_time = time.time()
                
                # Create optimized CoT-RAG instance
                cot_rag = create_optimized_cot_rag(
                    working_dir="./rag_cache_cot_optimized",
                    llm_model_func=self.llm_func,
                    embedding_func=self.embedding_func,
                    enable_cache=True
                )
                
                # Load data (with caching)
                load_and_insert_data(cot_rag, self.data_dir)
                
                # Query
                param = QueryParam(top_k=5)
                result = await cot_rag.query(query, param)
                
                end_time = time.time()
                
                # Measure memory after
                memory_after = self.measure_memory_usage()
                
                response_time = end_time - start_time
                memory_usage = memory_after - memory_before
                
                # Get cache stats
                stats = cot_rag.get_cot_stats()
                
                final_answer = ""
                confidence = 0.0
                total_steps = 0
                
                if result and hasattr(result, 'final_answer'):
                    final_answer = result.final_answer
                    confidence = getattr(result, 'confidence_score', 0.0)
                    total_steps = getattr(result, 'total_steps', 0)
                
                benchmark_result = BenchmarkResult(
                    test_name="Optimized CoT-RAG",
                    query=query,
                    response_time=response_time,
                    memory_usage=memory_usage,
                    cache_hits=stats.get('query_cache_hits', 0) + stats.get('cot_results_cache_size', 0),
                    cache_misses=stats.get('query_cache_misses', 0),
                    result_length=len(final_answer) if final_answer else 0,
                    success=bool(final_answer and len(final_answer.strip()) > 10)
                )
                
                results.append(benchmark_result)
                print(f"  Time: {response_time:.2f}s, Memory: {memory_usage:.1f}MB, Cache: {benchmark_result.cache_hits}H/{benchmark_result.cache_misses}M, Confidence: {confidence:.2f}, Steps: {total_steps}, Success: {benchmark_result.success}")
                
            except Exception as e:
                error_result = BenchmarkResult(
                    test_name="Optimized CoT-RAG",
                    query=query,
                    response_time=0.0,
                    memory_usage=0.0,
                    cache_hits=0,
                    cache_misses=0,
                    result_length=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                print(f"  Error: {e}")
        
        return results
    
    def analyze_results(self, results: List[BenchmarkResult]) -> Dict:
        """Analyze benchmark results and generate statistics"""
        if not results:
            return {}
        
        # Group by test name
        grouped_results = {}
        for result in results:
            if result.test_name not in grouped_results:
                grouped_results[result.test_name] = []
            grouped_results[result.test_name].append(result)
        
        analysis = {}
        
        for test_name, test_results in grouped_results.items():
            successful_results = [r for r in test_results if r.success]
            
            if successful_results:
                response_times = [r.response_time for r in successful_results]
                memory_usages = [r.memory_usage for r in successful_results]
                result_lengths = [r.result_length for r in successful_results]
                cache_hits = sum(r.cache_hits for r in successful_results)
                cache_misses = sum(r.cache_misses for r in successful_results)
                
                analysis[test_name] = {
                    'total_tests': len(test_results),
                    'successful_tests': len(successful_results),
                    'success_rate': len(successful_results) / len(test_results) * 100,
                    'avg_response_time': statistics.mean(response_times),
                    'median_response_time': statistics.median(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'avg_memory_usage': statistics.mean(memory_usages),
                    'avg_result_length': statistics.mean(result_lengths),
                    'total_cache_hits': cache_hits,
                    'total_cache_misses': cache_misses,
                    'cache_hit_rate': cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0
                }
            else:
                analysis[test_name] = {
                    'total_tests': len(test_results),
                    'successful_tests': 0,
                    'success_rate': 0,
                    'error': 'All tests failed'
                }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print formatted analysis results"""
        print("\n" + "="*80)
        print("BENCHMARK ANALYSIS RESULTS")
        print("="*80)
        
        for test_name, stats in analysis.items():
            print(f"\n{test_name}:")
            print("-" * 40)
            
            if 'error' in stats:
                print(f"  Error: {stats['error']}")
                continue
            
            print(f"  Tests: {stats['successful_tests']}/{stats['total_tests']} ({stats['success_rate']:.1f}% success)")
            print(f"  Response Time: {stats['avg_response_time']:.2f}s avg, {stats['median_response_time']:.2f}s median")
            print(f"  Response Time Range: {stats['min_response_time']:.2f}s - {stats['max_response_time']:.2f}s")
            print(f"  Memory Usage: {stats['avg_memory_usage']:.1f}MB avg")
            print(f"  Result Length: {stats['avg_result_length']:.0f} chars avg")
            
            if stats['total_cache_hits'] > 0 or stats['total_cache_misses'] > 0:
                print(f"  Cache Performance: {stats['total_cache_hits']} hits, {stats['total_cache_misses']} misses ({stats['cache_hit_rate']:.1f}% hit rate)")
    
    def save_results(self, results: List[BenchmarkResult], analysis: Dict, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_benchmark_{timestamp}.json"
        
        # Convert results to dict for JSON serialization
        results_dict = []
        for result in results:
            results_dict.append({
                'test_name': result.test_name,
                'query': result.query,
                'response_time': result.response_time,
                'memory_usage': result.memory_usage,
                'cache_hits': result.cache_hits,
                'cache_misses': result.cache_misses,
                'result_length': result.result_length,
                'success': result.success,
                'error_message': result.error_message
            })
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': results_dict,
            'analysis': analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    async def run_full_benchmark(self, save_results: bool = True) -> Tuple[List[BenchmarkResult], Dict]:
        """Run complete benchmark suite"""
        print("Starting RAG Performance Benchmark")
        print(f"Test queries: {len(self.test_queries)}")
        print(f"Data directory: {self.data_dir}")
        
        all_results = []
        
        # Benchmark original RAG
        original_rag_results = self.benchmark_original_rag(self.test_queries)
        all_results.extend(original_rag_results)
        
        # Benchmark optimized RAG
        optimized_rag_results = self.benchmark_optimized_rag(self.test_queries)
        all_results.extend(optimized_rag_results)
        
        # Benchmark original CoT-RAG
        original_cot_results = await self.benchmark_original_cot_rag(self.test_queries)
        all_results.extend(original_cot_results)
        
        # Benchmark optimized CoT-RAG
        optimized_cot_results = await self.benchmark_optimized_cot_rag(self.test_queries)
        all_results.extend(optimized_cot_results)
        
        # Analyze results
        analysis = self.analyze_results(all_results)
        
        # Print analysis
        self.print_analysis(analysis)
        
        # Save results
        if save_results:
            self.save_results(all_results, analysis)
        
        return all_results, analysis

async def main():
    """Main benchmark execution"""
    benchmark = RAGBenchmark(data_dir="./Crawl_data")
    
    try:
        results, analysis = await benchmark.run_full_benchmark(save_results=True)
        
        # Print performance comparison
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        if 'Original RAG' in analysis and 'Optimized RAG' in analysis:
            orig_time = analysis['Original RAG']['avg_response_time']
            opt_time = analysis['Optimized RAG']['avg_response_time']
            speedup = orig_time / opt_time if opt_time > 0 else 0
            print(f"RAG Speedup: {speedup:.2f}x faster ({orig_time:.2f}s → {opt_time:.2f}s)")
        
        if 'Original CoT-RAG' in analysis and 'Optimized CoT-RAG' in analysis:
            orig_time = analysis['Original CoT-RAG']['avg_response_time']
            opt_time = analysis['Optimized CoT-RAG']['avg_response_time']
            speedup = orig_time / opt_time if opt_time > 0 else 0
            print(f"CoT-RAG Speedup: {speedup:.2f}x faster ({orig_time:.2f}s → {opt_time:.2f}s)")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())