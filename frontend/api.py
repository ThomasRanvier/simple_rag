import streamlit as st
import requests
import time
from PyPDF2 import PdfReader


st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


st.title('Basic RAG with Mistral 7B')


def decode_stream(r):
    """Decode each stream yield as utf-8

    Args:
        r (generator of str): Generator of bytes strings

    Yields:
        str: Each decoded string
    """
    for chunk in r:
        yield chunk.decode('utf-8')


def get_pdf_text(pdf_docs):
    """Extract raw text from PDFs

    Args:
        pdf_docs (list of files): A list of PDFs files

    Returns:
        str: String of combined text from the PDFs files
    """
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


## Section to upload PDFs
with st.sidebar:
    st.subheader('Files')
    pdf_docs = st.file_uploader('Upload your PDFs here and click on \'Process\'', accept_multiple_files=True)
    ## Embed files through serving API
    if st.button('Process'):
        with st.spinner('Processing'):
            raw_text = get_pdf_text(pdf_docs)
            requests.post('http://simple-rag-backend:8080/embed', json={'content': raw_text})
    retrieve_k = st.slider('How many documents to retrieve?', min_value=1, max_value=5, value=3, step=1)


## Get user query
if query := st.chat_input('Enter your query'):
    with st.chat_message('user'):
        st.markdown(query)

    ## Get bot answer
    with st.chat_message('assistant'):
        start_time = time.time()
        with requests.post('http://simple-rag-backend:8080/search', json={'content': query, 'retrieve_k': retrieve_k}, stream=True) as r:
            st.write_stream(decode_stream(r))

        ## Indicate elapsed time for the request
        elapsed_time = time.time() - start_time
        st.markdown(f'---')
        st.markdown(f'*:red[- Time elapsed {elapsed_time:0.2f}s.]*')