import asyncio
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.prompt import get_system_prompt, build_rag_prompt
import os
import requests

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

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    RecursiveCharacterTextSplitter = None
    OllamaEmbeddings = None

def pdf_langchain_embedding(texts, model=None):
    """
    Embed PDF texts via LangChain splitter and OllamaEmbeddings.
    Each text is split into chunks and embedded.
    """
    if RecursiveCharacterTextSplitter is None or OllamaEmbeddings is None:
        raise ImportError("langchain_text_splitters or langchain_community not installed")
    # collect chunks
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    for text in texts:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
    # embed
    embedder = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    embs = embedder.embed_documents(all_chunks)
    return np.array(embs)


def openai_github_engine(prompt, api_key=None, endpoint=None, model=None):
    if api_key is None:
        api_key = os.environ.get("OPENAI_GITHUB_API_KEY")
    if endpoint is None:
        endpoint = "https://models.github.ai/inference"
    if model is None:
        model = "openai/gpt-4.1"
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Lỗi thư viện ")
    client = OpenAI(base_url=endpoint, api_key=api_key)
    import asyncio
    async def _call(prompt):
        completion = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    return asyncio.get_event_loop().run_until_complete(_call(prompt))

def openai_github_embedding(texts, api_key=None, endpoint=None, model=None):
    return [[0.0]*1536 for _ in texts]

def google_engine(prompt: str, api_key=None):
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Thiếu GOOGLE_API_KEY hoặc sai key")
    # Updated endpoint and model for Google Gemini/PaLM API
    # Use a supported model name for v1beta, e.g., gemini-2.0-flash-001
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "candidateCount": 1}
    }
    params = {"key": api_key}
    resp = requests.post(endpoint, headers=headers, params=params, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Google API error: {resp.status_code} {resp.text}")
    data = resp.json()
    # Extract the generated text from the response structure
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "[Google API] No output returned."


def google_embedding(texts, api_key=None):
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Thiếu GOOGLE_API_KEY hoặc sai key")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText"
    headers = {"Content-Type": "application/json"}
    payload = {"texts": texts}
    params = {"key": api_key}
    resp = requests.post(endpoint, headers=headers, params=params, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Google API trả về lỗi: {resp.status_code} {resp.text}")
    data = resp.json()
    return [item.get("embedding", []) for item in data.get("embeddings", [])]


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

def hugging_face_llm(prompt: str):
    """
    LLM function for Hugging Face that uses the multilingual E5 model.
    This is a simple implementation that returns the prompt as-is for now.
    """
    return prompt

def get_llm_func(engine: str):
    if engine == "google":
        api_key = get_api_key("google")
        return lambda prompt: google_engine(prompt, api_key=api_key)
    elif engine == "openai_github":
        token = get_api_key("openai_github")
        return lambda prompt: openai_github_engine(prompt, api_key=token)
    elif engine == "hugging_face":
        return lambda prompt: hugging_face_llm(prompt)
    else:
        return lambda prompt: prompt

def get_embedding_func(embed_engine: str):
    if embed_engine == "openai":
        api_key = get_api_key("openai")
        from rag import openai_embedding
        return lambda texts: openai_embedding(texts, api_key=api_key)
    elif embed_engine == "hugging_face":
        return lambda texts: hugging_face_embedding(texts)
    elif embed_engine == "google":
        api_key = get_api_key("google")
        return lambda texts: google_embedding(texts, api_key=api_key)
    elif embed_engine == "openai_github":
        token = get_api_key("openai_github")
        return lambda texts: openai_github_embedding(texts, api_key=token)
    else:
        raise ValueError(f"Unknown embed_engine: {embed_engine}")

