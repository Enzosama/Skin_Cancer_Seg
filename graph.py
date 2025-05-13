import os
import sys
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from rag.rag import RAG, QueryParam
from rag.llm import hugging_face_embedding
from cothought_rag import CoTRAG, load_and_insert_data_cotrag, cotrag_query
from native_rag import load_and_insert_data
# from dotenv import load_dotenv 
# load_dotenv()

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
    query = state["query"].lower()
    supported = False
    for topic, keywords in pathmap.items():
        if any(kw in query for kw in keywords):
            supported = True
            break
    if supported:
        semantic_result = f"Tìm thấy thông tin liên quan đến ung thư da từ Google Semantic Search."
        route = "node_2"
    else:
        semantic_result = "Không tìm thấy thông tin liên quan từ Google Semantic Search."
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
    print("---Node 2: RAG---")
    try:
        data_dir = os.path.join('Data')
        engine = state.get("engine", "google")
        llm_func = get_llm_func(engine)
        emb_func = hugging_face_embedding
        
        rag = RAG(
            working_dir='.',
            llm_model_func=llm_func,
            embedding_func=emb_func
        )
        load_and_insert_data(rag, data_dir)
        
        if rag.chunks:
            param = QueryParam(top_k=5)
            rag_result = await rag.query(state["query"], param)
            if rag_result and "\think" in state["query"]:
                route = "node_3"
            else:
                route = "node_4"
        else:
            rag_result = "Không có dữ liệu tham khảo."
            route = "node_4"
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
    print("---Node 3: CoT-RAG---")
    try:
        from rag.llm import get_llm_func, get_embedding_func
        engine = state.get("engine", "google")
        embed_engine = state.get("embed_engine", "hugging_face") if "embed_engine" in state else "hugging_face"
        top_k = state.get("top_k", 5)
        llm_func = get_llm_func(engine)
        emb_func = get_embedding_func(embed_engine)
        cotrag = CoTRAG(
            working_dir='.',
            llm_model_func=llm_func,
            embedding_func=emb_func
        )
        data_dir = os.path.join('Data')
        load_and_insert_data_cotrag(cotrag, data_dir)
        cot_rag_result = await cotrag_query(state["query"], cotrag, top_k=top_k)
        # Đảm bảo trả về chuỗi text
        if not isinstance(cot_rag_result, str):
            try:
                import numpy as np
                if isinstance(cot_rag_result, np.ndarray):
                    cot_rag_result = cot_rag_result.tolist()
                cot_rag_result = str(cot_rag_result)
            except Exception:
                cot_rag_result = str(cot_rag_result)
        final_result = cot_rag_result
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

