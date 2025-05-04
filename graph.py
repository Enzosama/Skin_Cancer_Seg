import os
import sys
import subprocess
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rag.rag import RAG, QueryParam
from rag.llm import hugging_face_embedding, hugging_face_llm

# Định nghĩa state cho graph
class AgentState(TypedDict):
    query: str
    rag_result: str
    llm_result: str
    use_rag: bool

# Node: gọi RAG
def log_debug(msg, state):
    with open('debug_graph.log', 'a', encoding='utf-8') as f:
        f.write(f'{msg}: {state}\n')

def rag_node(state: AgentState):
    print('--rag_node state--', state)
    log_debug('--rag_node state--', state)
    try:
        # Load file rag dataset
        data_file = os.path.join('Crawl_data', 'rag.csv')
        question = state['query']
        # Đọc dữ liệu
        df = pd.read_csv(data_file)
        docs = df['answer'].tolist()
        # Khởi tạo RAG
        rag = RAG(
            working_dir='.',
            llm_model_func=hugging_face_llm,
            embedding_func=hugging_face_embedding
        )
        rag.insert("\n\n".join(docs))
        param = QueryParam(top_k=5)
        rag_result = rag.query(question, param)
        use_rag = True if rag_result else False
    except Exception as e:
        rag_result = ''
        use_rag = False
        log_debug('--rag_node exception--', str(e))
    result = {
        **state,
        'rag_result': str(rag_result),
        'use_rag': use_rag
    }
    log_debug('--rag_node result--', result)
    return result

# Node: gọi LLM search
def search_llm_node(state: AgentState):
    print('--search_llm_node state--', state)
    log_debug('--search_llm_node state--', state)
    llm_result = f"Fake LLM search result for: {state['query']}"
    result = {
        **state,
        'llm_result': llm_result
    }
    log_debug('--search_llm_node result--', result)
    return result

# Logic điều hướng: nếu RAG ổn thì kết thúc, nếu không call search_llm
def decision_logic(state: AgentState):
    if state.get('use_rag'):
        return 'rag'
    else:
        return 'search_llm'

# Build graph
builder = StateGraph(AgentState)
builder.add_node('rag', rag_node)
builder.add_node('search_llm', search_llm_node)

builder.add_edge(START, 'rag')
builder.add_conditional_edges(
    'rag',
    decision_logic,
    {
        'rag': END,
        'search_llm': 'search_llm'
    }
)
builder.add_edge('search_llm', END)

compiled_graph = builder.compile()

if __name__ == '__main__':
    inputs = {
        'query': 'What are the early signs of skin cancer?',
        'rag_result': '',
        'llm_result': '',
        'use_rag': False
    }
    result = compiled_graph.invoke(inputs)
    print('Final result:', result)
