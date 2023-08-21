import logging

import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})


db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

# query = "details of second amendment"

# docs = db.similarity_search(query)
dic = db.get()

print(f'collections is {type(dic)}')
print(dic.keys())
print(len(dic.get('ids')))
print(len(dic.get('metadatas')))
