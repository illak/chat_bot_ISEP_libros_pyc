import os

from langchain_community.document_loaders import PDFPlumberLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from langchain_core.runnables import RunnablePassthrough
import streamlit.components.v1 as components



SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# read all pdf files and return text
@st.cache_resource
def load_pdfs():

    directory = "libros_pyc"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=400) # chunk_overlap 1000

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    # iterate over files in
    # that directory
    for index, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):

            print(f)
            loader = PDFPlumberLoader(f)

            data = loader.load()

            for document in data:
                # Unifico división silábica por salto de línea
                document.page_content = re.sub(r'-\n', '', document.page_content)
                document.page_content = re.sub(r'(?<!\.)\n', ' ', document.page_content)


            splits = splitter.split_documents(data)

            #vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
            if index == 0:
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            else:
                db_i = FAISS.from_documents(documents=splits, embedding=embeddings)
                vectorstore.merge_from(db_i)

            print(vectorstore.index.ntotal)


    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )
    # Devolvemos retriever
    return retriever


# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def debug_prompt(x):
    print(x)
    return x


# testing
def get_conversational_chain(retriever):

    prompt_template = """Responder a la pregunta del usuario lo más detallademente posible. Si la respuesta no se encuentra
en el contexto provisto responda que no encontró información en dicho contexto, no devuelva una respuesta incorrecta.\n\n
-------------------------------
Contexto: {context}\n
-------------------------------\n

Pregunta: {question}\n

Respuesta:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   maxOutputTokens=2500,
                                   safety_settings=SAFETY_SETTINGS
                                   )
    
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    
    #chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | debug_prompt
        | model 
        | StrOutputParser()
    )

    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Haga preguntas..."}]


def user_input(user_question, retriever):

    chain = get_conversational_chain(retriever)

    response = chain.invoke(user_question)
    
    print(response)
    return response


def main(): 
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="📚"
    )

    # Custom HTML/CSS for the banner
    custom_html = """
    <div class="banner">
        <img src="https://isep-cba.edu.ar/web/wp-content/uploads/2023/05/encabezado_ColeccionPyC-1024x474.jpg" alt="Banner Image">
    </div>
    <style>
        .banner {
            width: 100%;
            height: auto;
            overflow: contain;
        }
        .banner img {
            width: 100%;
            object-fit: scale-down;
        }
    </style>
    """
    # Display the custom HTML
    #components.html(custom_html)
    st.image('./imgs/pyc_header.jpg', caption='Colección Pedagogía y Cultura')






    # Cargamos los libros de PYC al INICIO
    with st.spinner("Procesando PDFs..."):
        retriever = load_pdfs()

    # Sidebar for uploading PDF files
    _ = """with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")"""


    # Main content area for displaying chat messages
    st.header("Chateando con los libros de Pedagogía y Cultura (ISEP) 📚")
    st.write("Bienvenido al chat!")
    st.sidebar.button('Limpiar chat', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Escriba alguna pregunta sobre los libros"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Estoy pensando..."):
                #response = st.write_stream(user_input(prompt))
                response = user_input(prompt, retriever)
                placeholder = st.empty()
                full_response = response
                #full_response = ''
                #for item in response['output_text']:
                #    full_response += item
                #    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
