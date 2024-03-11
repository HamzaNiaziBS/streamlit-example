from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from JSONLoaderAPI import JSONLoaderAPI
import streamlit as st
from streamlit_chat import message
from langchain.callbacks import StreamlitCallbackHandler


# LLM
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema.document import Document
from langchain.vectorstores.pgvector import PGVector
import requests


from pprint import pprint
from urllib.request import urlopen
from glob import glob
from PIL import Image
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

import streamlit as st
import os



def load_chain():

    CONNECTION_STRING = st.secrets["CONNECTION_STRING"]
    COLLECTION_NAME = "BS_CRM_OPPORTUNITIES_NOTES"

    def metadata_func(record: dict, metadata: dict) -> dict:

        metadata["opportunity_code"] = record.get("opportunity_code")
        metadata["amount"] = record.get("amount")
        metadata["opportunity_name"] = record.get("opportunity_name")
        metadata["added_by"] = record.get("added_by")
        metadata["added_on"] = record.get("added_on")
        metadata["problem_statement"] = record.get("problem_statement")
        metadata["sales_stage"] = record.get("sales_stage")
        metadata["id"] = record.get("id")
        return metadata

    url = "https://bigspark.world/api/method/opps-notes"

    payload = {}
    headers = {
      'Authorization': st.secrets["Authorization"]
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    
    loader = JSONLoaderAPI(
            json_object =  response.text,
        jq_schema='.message[]',
        content_key='note_content',
        metadata_func=metadata_func,
    )

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    persist_directory = 'opps'


    # ids = [(doc.metadata["id"]) for doc in all_splits]
    # print(ids)
    # unique_ids = list(set(ids))

    vectorstore = PGVector.from_documents(
        embedding=GPT4AllEmbeddings(),
        documents=all_splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection = True
    )


    template = """
    Summarize the main themes in these retrieved docs and add extra information from metadata:
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )



    llm = Ollama(base_url="http://localhost:11434",
                 model="orca-mini",
                 verbose=True,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    # QA chain

    # Define a class for ContextualRetriever that inherits VectorStoreRetriever
    class ContextualRetriever(VectorStoreRetriever):
        # Override the '_get_relevant_documents' method
        def _get_relevant_documents(self, query: str, *, run_manager):
            # Call the parent's '_get_relevant_documents' method
            docs = super()._get_relevant_documents(query, run_manager=run_manager)
            return [self.format_doc(doc) for doc in docs]

        # Method to format the document
        def format_doc(self, doc: Document) -> Document:
            # Format the page content of the document
            doc.page_content =  f"Opportunity name: {doc.metadata['opportunity_name']}, opportunity code: {doc.metadata['opportunity_code']}, added by: {doc.metadata['added_by']}, added on: {doc.metadata['added_on']}, problem statement: {doc.metadata['problem_statement']}, note content {doc.page_content}, opportunity  amount: {int(doc.metadata['amount']):d}, sales stage: {doc.metadata['sales_stage']}"
            return doc

    # Create a RetrievalQA object from a chain type
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=ContextualRetriever(vectorstore=vectorstore),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,

        chain_type="stuff",
    )

    # Define the question
    # question = "Who is shaine going to meet? tell memore about the opportunity"
    # Get the result by invoking the qa_chain
    # result = qa_chain({"query": question})
    return qa_chain


# From here down is all the StreamLit UI.

st.set_page_config(page_title="Bigspark CRM", page_icon=":robot:")

#displaying the image on streamlit app
image = Image.open('./images/bigspark.jpg')
st.image(image)

st.header("Bigspark CRM LLM")
st_callback = StreamlitCallbackHandler(st.container())


def get_text():
    input_text = st.text_input("Enter your question about Opportunities?", key="input")
    return input_text


user_input = get_text()
submitted = st.button('Ask Question')

if submitted:
        chain = load_chain()
        output = chain({"query": user_input}, callbacks=[st_callback])
        print(output["source_documents"])

        st.info(output)

