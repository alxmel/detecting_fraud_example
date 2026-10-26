import streamlit as st

##### Import agent from sibling directory
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
#####

from agent.agent import create_agent

# Absolute path so the image loads from any launch directory
AGENT_IMAGE = os.path.join(os.path.dirname(__file__), "image", "logo.jpeg")

# Initialize the agent once; it persists across reruns via session_state
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Accept user input; read first so the home image can hide on the first message (the chat box stays pinned to the bottom)
prompt = st.chat_input("Welcome to the Anomaly Detector!")

# Home page image, shown until the conversation starts
if not st.session_state.messages and not prompt:
    st.image(AGENT_IMAGE)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = AGENT_IMAGE if message["role"] == "assistant" else None   # None keeps the default user icon
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})   # Add user message to chat history
    with st.chat_message("user"):   # Display user message in chat message container
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant", avatar=AGENT_IMAGE):
        with st.spinner("..."):
            try:
                response = str(st.session_state.agent(prompt))
            except Exception as error:
                response = f"Sorry, something went wrong: {error}"
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

