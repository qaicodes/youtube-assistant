import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label="Ask me about the video",
            max_chars=50,
            key="query"
        )
        submit_button = st.form_submit_button(label="SUBMIT")

if query and youtube_url:
    db = lch.vector_db_from_youtube(youtube_url)
    response = lch.get_response(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))