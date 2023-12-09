from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def vector_db_from_youtube(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response(db, query, k=4):
    # text-davinci can handle upto 4097 tokens
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm= OpenAI(model='text-davinci-003')
    prompt=PromptTemplate(
        input_variables=['question', 'docs'],
        template="""
            You are an helpful Youtube assistant that can answer questions about videos on the basis of it's transcript.

            Answer the following question: {question}
            By searching the following video transcript: {docs}

            If you feel like you don't have enough information to ask the question, just say "I don't know".

            Your answers should be detailed.
            """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs = docs_page_content)
    response = response.replace("\n", "")
    return response