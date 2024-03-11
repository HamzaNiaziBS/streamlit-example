
import random
import streamlit as st

"""
# Welcome to Streamlit Chatbot!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
"""

# predefined responses
RESPONSES = {
    "hello": ["Hello!", "Hi there!", "Hey! How can I help you today?"],
    "how are you": ["I'm a bot, I don't have feelings, but thanks for asking!", "I'm always ready to assist you!"],
    "bye": ["Goodbye!", "See you later!", "Have a nice day!"]
}

def get_response(user_input):
    for keyword, responses in RESPONSES.items():
        if keyword in user_input.lower():
            return random.choice(responses)
    return "I'm sorry, I didn't understand that. Could you rephrase?"

with st.echo(code_location='below'):
    user_input = st.text_input("Ask something: ")
    if user_input:
        st.write(get_response(user_input))
