import streamlit as st
from Model.MistralAgent import Model


def chat(model):
    st.title("Research Assistant Chatbot")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your message:", "")

    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")

        bot_response = model.response(user_input)
        st.session_state.chat_history.append(f"Bot: {bot_response}")

    for message in st.session_state.chat_history:
        st.write(message)


if __name__ == "__main__":
    model = Model("Parag")
    chat(model)
