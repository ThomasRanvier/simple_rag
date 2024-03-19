from utils.lm import *

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import logging
import os


ROOT_LEVEL = os.environ.get('PROD', 'INFO')

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'},
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'level': ROOT_LEVEL,
            'handlers': ['default'],
            'propagate': False,
        },
        'uvicorn.error': {
            'level': 'DEBUG',
            'handlers': ['default'],
        },
        'uvicorn.access': {
            'level': 'DEBUG',
            'handlers': ['default'],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


logger = logging.getLogger(__name__)
app = FastAPI()


class Query(BaseModel):
    content: str
    retrieve_k: int


class Pdfs(BaseModel):
    content: str


## Initiate models
logger.info(f'Torch: {torch.__version__}')
logger.info(f'Cuda available: {torch.cuda.is_available()}')
logger.info('Loading LLM and Tokenizer')
llm, tokenizer = get_llm_and_tokenizer()
logger.info('Loading Embedding Model')
embedding_model = get_embedding_model(model_name='dangvantuan/sentence-camembert-large')
logger.info('Done')
vectorstore = None


def query_llm_api(query, retrieve_k):
    """Query the LLM using the RAG paradigm

    Args:
        query (str): The query to ask the LLM
        retrieve_k (int): How many documents to retrieve for RAG

    Yields:
        str: Each word yielded from the LLM
    """
    logger.info('- Querying most similar docs')
    docs = vectorstore.similarity_search(query, k=retrieve_k)
    context = ''
    for doc in docs:
        context += doc.dict()['page_content'] + '\n\n'
    logger.info('- Context:')
    logger.info(context)
    logger.info('- Querying LLM with docs context')
    for word in rag_streaming_query(context, query, llm, tokenizer):
        yield word


@app.post('/embed')
def embed(pdfs: Pdfs):
    """Embeds the given files in a vector database

    Args:
        pdfs (Pdfs): Contains the raw text from PDFs
    """
    global vectorstore
    logger.info('- Creating vector database from pdfs data')
    vectorstore = create_vectorstore(split_text(pdfs.content), embedding_model)


@app.post('/search')
async def search(query: Query):
    """Queries the LLM using the RAG paradigm

    Args:
        query (Query): Contains the text of the user query

    Returns:
        StreamingResponse: The streaming output of the LLM to answer the query
    """
    return StreamingResponse(query_llm_api(query.content, query.retrieve_k), media_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)