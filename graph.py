import os
import sys
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from rag.rag import RAG, QueryParam
from rag.llm import hugging_face_embedding
from cothought_rag import CoTRAG, load_and_insert_data_cotrag, cotrag_query
from native_rag import load_and_insert_data
from optimized_rag import OptimizedRAG, create_optimized_rag
from optimized_cot_rag import OptimizedCoTRAG, create_optimized_cot_rag
from dotenv import load_dotenv 
load_dotenv()
from groq import Groq

def get_api_key(service: str):
    env_map = {
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai_github": "OPENAI_GITHUB_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "hugging_face": "HUGGING_FACE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key = env_map.get(service)
    if not key:
        raise ValueError(f"No API key mapping for service {service}")
    value = os.environ.get(key)
    if not value:
        print(f"Missing API key for {service} ({key})", file=sys.stderr)
        sys.exit(1)
    return value

class AgentState(TypedDict):
    query: str
    semantic_result: str
    rag_result: str
    cot_rag_result: str
    final_result: str
    route: str
    engine: str

# Node 1: Semantic search 
pathmap = {
    "skin_cancer": [
        "nv", "mel", "bkl", "bcc", "vasc", "akiec", "df", "ung thư", "ung thư da", "nốt ruồi sắc tố", "ung thư hắc tố", "tổn thương lành tính giống tăng sản sừng", "ung thư biểu mô tế bào đáy", "u hạt mủ và chảy máu", "dày sừng quang hóa và ung thư biểu mô tại chỗ", "u sợi da", "ung thư biểu mô tế bào vảy", "skin cancer", "melanocytic nevi", "melanoma", "benign keratosis-like lesions", "basal cell carcinoma", "pyogenic granulomas and hemorrhage", "Actinic keratoses and intraepithelial carcinoma", "dermatofibroma", "squamous cell carcinoma"
    ]
}
def node_1(state: AgentState):
    print("---Node 1: Semantic Search ---")
    query = state["query"]
    if "\\think" in query:
        semantic_result = "Chuyển trực tiếp sang CoT-RAG do truy vấn chứa từ khóa \\think."
        route = "node_3"
        return {
            **state,
            "semantic_result": semantic_result,
            "route": route
        }
    api_key = get_api_key("groq")
    client = Groq(api_key=api_key)
    prompt = (
        "Bạn là chuyên gia y tế. Hãy xác định liệu câu hỏi sau có liên quan đến chủ đề ung thư da không. "
        "Nếu liên quan, trả lời 'SUPPORTED'. Nếu không, trả lời 'UNSUPPORTED'.\n"
        f"Câu hỏi: {query}\n"
        "Chỉ trả lời 'SUPPORTED' hoặc 'UNSUPPORTED'."
    )
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y tế về chủ đề ung thư da."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        result = completion.choices[0].message.content.strip().upper()
        if "SUPPORTED" in result:
            semantic_result = "Tìm thấy thông tin liên quan đến ung thư da từ Groq Semantic Search."
            route = "node_2"
        else:
            semantic_result = "Không tìm thấy thông tin liên quan từ Groq Semantic Search."
            route = "node_4"
    except Exception as e:
        print(f"[ERROR] {e}")
        semantic_result = "Lỗi khi gọi Groq Semantic Search."
        route = "node_4"
    return {
        **state,
        "semantic_result": semantic_result,
        "route": route
    }

# Hàm chọn llm_model_func dựa trên engine
from rag.llm import hugging_face_llm

def get_llm_func(engine: str):
    if engine == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            return lambda prompt: hugging_face_embedding([prompt])[0]
        return hugging_face_embedding
    else:
        # fallback: echo prompt
        return lambda prompt: prompt

# Node 2 RAG
async def node_2(state: AgentState):
    print("---Node 2: Optimized RAG---")
    try:
        data_dir = os.path.join('Data')
        engine = state.get("engine", "google")
        llm_func = get_llm_func(engine)
        emb_func = hugging_face_embedding
        
        # Create optimized RAG instance
        rag = create_optimized_rag(
            working_dir='.',
            llm_model_func=llm_func,
            embedding_func=emb_func,
            enable_cache=True
        )
        
        # Load and insert data (with caching)
        load_and_insert_data(rag, data_dir)
        
        if rag.chunks:
            param = QueryParam(top_k=5)
            rag_result = await rag.query(state["query"], param)
            if (
                (rag_result is None or rag_result == "" or rag_result == "Không có dữ liệu tham khảo.")
                and (state.get("semantic_result", "").upper().find("UNSUPPORTED") != -1 or state.get("semantic_result", "").find("không tìm thấy thông tin liên quan".lower()) != -1)
            ):
                route = "node_4"
            elif "\\think" in state["query"]:
                route = "node_3"
            else:
                route = "node_3"
        else:
            rag_result = "Không có dữ liệu tham khảo."
            if state.get("semantic_result", "").upper().find("UNSUPPORTED") != -1 or state.get("semantic_result", "").find("không tìm thấy thông tin liên quan".lower()) != -1:
                route = "node_4"
            else:
                route = "node_3"
    except Exception as e:
        rag_result = f"Lỗi RAG: {e}"
        route = "node_4"
    return {
        **state,
        "rag_result": str(rag_result),
        "route": route
    }

# Node 3: Sử dụng CoT-RAG
async def node_3(state: AgentState):
    print("---Node 3: Optimized CoT-RAG---")
    try:
        from rag.llm import get_llm_func, get_embedding_func
        engine = state.get("engine", "google")
        embed_engine = state.get("embed_engine", "hugging_face") if "embed_engine" in state else "hugging_face"
        top_k = state.get("top_k", 5)
        llm_func = get_llm_func(engine)
        emb_func = get_embedding_func(embed_engine)
        
        # Create optimized CoT-RAG instance
        cotrag = create_optimized_cot_rag(
            working_dir='.',
            llm_model_func=llm_func,
            embedding_func=emb_func,
            max_reasoning_steps=5,
            confidence_threshold=0.7,
            enable_cache=True
        )
        
        data_dir = os.path.join('Data')
        load_and_insert_data_cotrag(cotrag, data_dir)
        cot_rag_result = await cotrag_query(state["query"], cotrag, top_k=top_k, detailed=False)
        
        # Handle both old string format and new CoTResult format for backward compatibility
        if isinstance(cot_rag_result, str):
            final_result = cot_rag_result
        else:
            # If it's a CoTResult object, extract the final answer
            try:
                from cot_rag import CoTResult
                if hasattr(cot_rag_result, 'final_answer'):
                    final_result = cot_rag_result.final_answer
                    if hasattr(cot_rag_result, 'confidence_score') and hasattr(cot_rag_result, 'total_steps'):
                        print(f"[NODE_3] Confidence: {cot_rag_result.confidence_score:.2f}, Steps: {cot_rag_result.total_steps}")
                elif hasattr(cot_rag_result, 'answer'):
                    final_result = cot_rag_result.answer
                else:
                    final_result = str(cot_rag_result)
            except Exception:
                # Fallback to string conversion
                final_result = str(cot_rag_result)
        route = "end"
    except Exception as e:
        cot_rag_result = f"Lỗi CoT-RAG: {e}"
        final_result = cot_rag_result
        route = "node_4"
    return {
        **state,
        "cot_rag_result": cot_rag_result,
        "final_result": final_result,
        "route": route
    }

# Node 4: Trả về "Không hỗ trợ"
def node_4(state: AgentState):
    print("---Node 4: Not related to Skin cancer---")
    return {
        **state,
        "final_result": "Không hỗ trợ",
        "route": "end"
    }

def decision_node(state: AgentState) -> Literal["node_2", "node_4"]:
    return state.get("route", "node_4")

def decision_node_2(state: AgentState) -> Literal["node_3", "node_4"]:
    return state.get("route", "node_4")

def decision_node_3(state: AgentState) -> Literal["end", "node_4"]:
    return state.get("route", "end")

builder = StateGraph(AgentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_node("node_4", node_4)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decision_node, {"node_2": "node_2", "node_4": "node_4"})
builder.add_conditional_edges("node_2", decision_node_2, {"node_3": "node_3", "node_4": "node_4"})
builder.add_conditional_edges("node_3", decision_node_3, {"end": END, "node_4": "node_4"})
builder.add_edge("node_4", END)

compiled_graph = builder.compile()

if __name__ == "__main__":
    import asyncio
    inputs = {
        "query": "What causes melanoma?",
        "semantic_result": "",
        "rag_result": "",
        "cot_rag_result": "",
        "final_result": "",
        "route": "",
    }
    async def main():
        result = await compiled_graph.ainvoke(inputs)
        print("Kết quả cuối cùng:", result)
    asyncio.run(main())

class CustomRAG(RAG):
    def __init__(self, working_dir, llm_model_func, embedding_func):
        super().__init__(working_dir=working_dir, llm_model_func=llm_model_func, embedding_func=embedding_func)

