# Local RAG for course documents
# https://blog.gopenai.com/retrieval-augmented-generation-rag-using-llama-2-chromadb-together-ai-for-retrieval-qa-80e7045a6c79

import os # interact with shell 
import re # regular expressions

# together.ai hosts the LLM that we're using here 
import together

# langchain
from langchain.vectorstores import Chroma # stores the embeddings generated locally
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader # loads pdfs
from langchain.document_loaders import DirectoryLoader # searches entire directory for mass file ingestion
from langchain_community.embeddings import HuggingFaceInstructEmbeddings # get embeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma # Supplying a persist_directory will store the embeddings on disk
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

# vector store
import chromadb

# embeddings 
# https://instructor-embedding.github.io/
from InstructorEmbedding import INSTRUCTOR

# llama
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext

import textwrap # process text from llm to make it readable
import logging
from typing import Any, Dict, List, Mapping, Optional

# make sure you only pass certain data to the llm iteslf
from pydantic.v1 import Extra, Field, root_validator

# where are we?
os.getcwd()

# set working directory
os.chdir("/Users/stevenrashin/stats-material/")

# constants
persist_directory = '/Users/stevenrashin/stats-material/output/' # this is where we store the local embeddings

##### PUT TOGETHER.API KEY HERE
together.api_key = ""
os.environ["TOGETHER_API_KEY"] = ""

# currently works though pdf to text not great for pdfs
#https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
loader = DirectoryLoader('/Users/stevenrashin/stats-material', glob="*.pptx", use_multithreading=True)

documents = loader.load()
# access data using the .page_content[begin character: end character] as in:
# documents[0].page_content[:40]

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


# Embed and store the texts
#https://huggingface.co/hkunlp/instructor-xl
#https://arxiv.org/abs/2212.09741
#https://python.langchain.com/docs/integrations/vectorstores/chroma
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cpu"})

## Here is the nmew embeddings being used
embedding = instructor_embeddings

# This function creates and stores embeddings

# create empty databse that you'll populate 
#client = chromadb.PersistentClient(path="output/")
# In specfying the persist_directory argument, make sure the file path points to the filename of the sqlite database itself i.e. '/Users/stevenrashin/stats-material/output/chroma.sqlite3'
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory
                                 )

# see all the LLM models that together has
models = together.Models.list()
model_names = [model_dict['name'] for model_dict in models]
model_names[:10]

# search for the model you want
pattern = '[Ll]ama' and 'hf'
for element in model_names:
    if re.search(pattern, element):
        print(f"'{element}' matches the pattern.")

# set a class for the LLM
class TogetherLLM(LLM):
    """Together large language models.  The strings are default values that will be used unless specified """

    model: str = "togethercomputer/llama-2-13b-chat"
    """model endpoint to use"""
    """can replace with any model in together.ai's index that will print above"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text

# k = number of documents to return (will always return 5 if k = 5.  More takes longer but better answer)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# https://deepchecks.com/glossary/llm-parameters/#:~:text=One%20intriguing%20parameter%20within%20LLMs,creativity%2C%20of%20the%20AI's%20responses.
# temperature regulates randomness
llm = TogetherLLM(
    model= "togethercomputer/llama-2-13b-chat",
    temperature = 0.1,
    max_tokens = 1024
)

# create the chain to answer questions
#The valid values for the chain_type parameter in the from_chain_type method in the LangChain framework have been updated to include "stuff", "map_reduce", "refine", and "map_rerank".

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query = '''Why do we use standard deviation?
'''

llm_response = qa_chain(query)
process_llm_response(llm_response)
