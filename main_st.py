"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = Ollama(base_url="http://52.56.89.253",
             model="orca-mini",
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Bigspark CRM LLM Demo", page_icon=":robot:")
st.header("Bigspark CRM LLM Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Who is ___ going to meet?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
