import asyncio
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer

# Optionally instantiate sentence-transformers model for compatibility
st_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# HuggingFace Multilingual-E5-Large-Instruct embedding engine
_hf_tokenizer = None
_hf_model = None
def get_hf_e5_model():
    global _hf_tokenizer, _hf_model
    if _hf_tokenizer is None or _hf_model is None:
        _hf_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
        _hf_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
    return _hf_tokenizer, _hf_model

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

from rag.prompt import get_system_prompt, build_rag_prompt

def get_detailed_instruct(task_description: str, query: str) -> str:
    # Use system prompt from prompt.py for consistency
    system = get_system_prompt(task_description)
    return build_rag_prompt("", query, system)

# Embedding function for hugging_face engine
def hugging_face_embedding(texts, task=None):
    if task is None:
        task = "Given a web search query, retrieve relevant passages that answer the query"
    tokenizer, model = get_hf_e5_model()
    input_texts = [get_detailed_instruct(task, t) for t in texts]
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# Load Multilingual-E5-Large-Instruct model once
_model = None
def get_e5_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    return _model

# Dummy LLM functions (replace with real implementations as needed)
async def google_complete(prompt: str):
    return "[Google LLM] Answer to: " + prompt

async def llama_complete(prompt: str):
    return "[LLaMA LLM] Answer to: " + prompt

async def hugging_face_llm(prompt: str):
    return "[HuggingFace LLM] Answer to: " + prompt

async def openai_embedding(texts, model=None):
    model = get_e5_model()
    # E5 expects "query: ..." or "passage: ..."
    processed = [f"query: {t}" if isinstance(t, str) else str(t) for t in texts]
    emb = model.encode(processed, convert_to_numpy=True, show_progress_bar=False)
    return emb

async def google_embedding(texts, model=None):
    model = get_e5_model()
    processed = [f"query: {t}" if isinstance(t, str) else str(t) for t in texts]
    emb = model.encode(processed, convert_to_numpy=True, show_progress_bar=False)
    return emb

async def groq_embedding(texts, model=None):
    model = get_e5_model()
    processed = [f"query: {t}" if isinstance(t, str) else str(t) for t in texts]
    emb = model.encode(processed, convert_to_numpy=True, show_progress_bar=False)
    return emb
