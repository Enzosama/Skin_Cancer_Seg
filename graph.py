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
from cot_rag.rag import CoTResult
from dotenv import load_dotenv 
load_dotenv()
from groq import Groq
import asyncio

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
    # Check if the query contains \think at the beginning or end of the sentence
    if query.strip().startswith("\\think") or query.strip().endswith("\\think"):
        semantic_result = "Chuyển trực tiếp sang CoT-RAG do truy vấn chứa từ khóa \\think ở đầu hoặc cuối câu."
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
            # When status is UNSUPPORTED, transition to node 4
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

# Node 2: RAG
def node_2(state: AgentState):
    """RAG"""
    try:
        rag = create_optimized_rag(
            working_dir="./rag_cache",
            llm_model_func=None,
            embedding_func=hugging_face_embedding,
            enable_cache=True,
            use_sqlite=True
        )
        load_and_insert_data(rag, "./Data")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(rag.query(state["query"]))
        state["rag_result"] = result
        if "\think" in state["query"]:
            state["final_result"] = result
        else:
            state["final_result"] = result
        return state
    except Exception as e:
        print(f"Error in node_2: {e}")
        state["final_result"] = "Không hỗ trợ"
        return state

# Node 3: Sử dụng CoT-RAG
def node_3(state: AgentState):
    """CoT-RAG"""
    try:
        cot_rag = create_optimized_cot_rag(
            working_dir="./cot_rag_cache",
            llm_model_func=None,
            embedding_func=hugging_face_embedding,
            enable_cache=True,
            use_sqlite=True
        )
        load_and_insert_data_cotrag(cot_rag, "./Data")
        # Run the async query in a synchronous context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(cot_rag.query(state["query"]))
        if isinstance(result, str):
            state["final_result"] = result
            state["cot_rag_result"] = result
        elif isinstance(result, CoTResult):
            state["final_result"] = result.final_answer
            state["cot_rag_result"] = result.final_answer
        else:
            state["final_result"] = str(result)
            state["cot_rag_result"] = str(result)
        return state
    except Exception as e:
        print(f"Error in node_3: {e}")
        state["final_result"] = "Không hỗ trợ"
        state["cot_rag_result"] = "Không hỗ trợ"
        return state

# Node 4: Trả về "Không hỗ trợ"
def node_4(state: AgentState):
    print("---Node 4: Not related to Skin cancer---")
    return {
        **state,
        "final_result": "Không hỗ trợ",
        "route": "end"
    }

def decision_node(state: AgentState) -> Literal["node_2", "node_4"]:
    # For node 1, when the status is UNSUPPORTED, transition to node 4
    route = state.get("route", "node_4")
    semantic_result = state.get("semantic_result", "")
    
    # Transition to node_4 if semantic search result is UNSUPPORTED or if there was an error
    if route == "node_4" or semantic_result == "UNSUPPORTED" or "Lỗi" in semantic_result:
        return "node_4"
    # Otherwise, transition to node_2 (RAG)
    return "node_2"

def decision_node_2(state: AgentState) -> Literal["node_3", "end"]:
    route = state.get("route", "end")
    query = state.get("query", "")
    
    # Check if query contains \think to use CoT-RAG
    if route == "node_3" or "\\think" in query:
        return "node_3"
    
    return "end"

def decision_node_3(state: AgentState) -> Literal["end"]:
    # Node 3 should always transition to end, never to node 4
    return "end"

builder = StateGraph(AgentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_node("node_4", node_4)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decision_node, {"node_2": "node_2", "node_4": "node_4"})
builder.add_conditional_edges("node_2", decision_node_2, {"node_3": "node_3", "end": END})
builder.add_conditional_edges("node_3", decision_node_3, {"end": END})
builder.add_edge("node_4", END)

compiled_graph = builder.compile()

if __name__ == "__main__":
    inputs = {
        "query": "\think What causes melanoma?",
        "semantic_result": "",
        "rag_result": "",
        "cot_rag_result": "",
        "final_result": "",
        "route": "",
    }
    # Use invoke instead of ainvoke since we've made the nodes synchronous
    result = compiled_graph.invoke(inputs)
    print("Kết quả cuối cùng:", result)

class CustomRAG(RAG):
    def __init__(self, working_dir, llm_model_func, embedding_func):
        super().__init__(working_dir=working_dir, llm_model_func=llm_model_func, embedding_func=embedding_func)
