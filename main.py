# Script para la nueva app

import ast
import hmac
import json
import openai
import os
import streamlit as st
from jsonbin_funs_v3 import load_json
# from jsonbin_funs_v3 import get_remain_obs, add_registro, load_json, descontar_obs, ultimas_preguntas
# from decorador_costos import decorador_costo
from streamlit_chat import message
import qdrant_client
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant
from PyPDF2 import PdfReader
from qdrant_client.http import models

# streamlit run /Users/allan/Documents/Git/chateaconeldoc/main.py

api_key = st.secrets["OPENAI_API_KEY"]
X_Master_Key = st.secrets["X_MASTER_KEY"]
bin_id_usuarios = st.secrets["BIN_ID_USUARIOS"]
os.environ["OPENAI_API_KEY"] = api_key

info_usuarios = load_json(bin_id_usuarios,X_Master_Key)

usuarios = [key for key, value in info_usuarios.items()]

def get_pdf_text(pdf_docs):

  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
  return text

def get_text_chunks(text):

  text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
  )

  chunks = text_splitter.split_text(text)

  return chunks

def authenticate_user_hoy():
  if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

  if st.session_state.authenticated:
    return True

  user_placeholder = st.empty()
  password_placeholder = st.empty()
  button_placeholder = st.empty()

  st.session_state.usuario = user_placeholder.text_input(label="Usuario")
  st.session_state.password = password_placeholder.text_input(label="Contraseña", type="password")

  if button_placeholder.button('Login'):
      if st.session_state.usuario in usuarios and hmac.compare_digest(
        st.session_state.password,
        info_usuarios[st.session_state.usuario]["psswd"],
      ):
        st.session_state.authenticated = True

        user_placeholder.empty()
        password_placeholder.empty()
        button_placeholder.empty()

        return True
      elif st.session_state.usuario and st.session_state.password:
        st.error("Por favor ingresa un usuario y contraseña correctos.")
        return False

def handle_user_input(user_question):
  st.session_state.conversation.append(
    {
      "question": user_question  
    }
  )
  st.session_state.conversation.append(
    {
      "answer": st.session_state.response
    }
  )
  for i in range(len(st.session_state.conversation) - 1, -1, -2):
    if i-1 >= 0:
      question = st.session_state.conversation[i-1]["question"]
      message(question,is_user=True,key=str(i) + '_user')

    answer = st.session_state.conversation[i]["answer"]
    message(answer,is_user=False,key=str(i) + '_ai')
    
def obtener_colecciones(qdrant_api_key,qdrant_host):

  client = qdrant_client.QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key,
  )

  colecciones = client.get_collections()

  for item in colecciones:
    nombres_colecciones = [coleccion.name for coleccion in item[1]]

  return nombres_colecciones  

def create_vectorstore(text_chunks, coleccion, qdrant_api_key, qdrant_host,openai_api_key):

  client = qdrant_client.QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key,
  )

  vectors_configuration = models.VectorParams(
    size=1536,
    distance=models.Distance.COSINE
  )

  client.recreate_collection(
    collection_name=coleccion,
    vectors_config=vectors_configuration,
  )
  # Aquí se termina de crear la colección

  # Se carga el vectorstore
  embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
  )

  vectorstore = Qdrant(
    client=client,
    collection_name=coleccion,
    embeddings=embeddings
  )

  # Se agregan los embeddings de los documentos al vectorstore
  vectorstore.add_texts(
    text_chunks,
  )

  print("------------------------------------------")
  print("El vectorstore ha sido generado con éxito.")
  print("------------------------------------------")
    
def load_vectorstore(qdrant_api_key,qdrant_host,qdrant_collection_name,openai_api_key):

  client = qdrant_client.QdrantClient(
    qdrant_host,
    api_key=qdrant_api_key
  )

  vectorstore = Qdrant(
    client=client,
    collection_name=qdrant_collection_name,
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
  )

  return vectorstore

def get_conversation_chain(user_question,openai_api_key,coleccion,instrucciones = "Eres un experto en analizar preguntas acerca del texto compartido y obtener conclusiones acertadas."):

  general_system_template = "Dado el context"+instrucciones+r"""
  ----
  {context}
  ----
  """
  general_user_template = "Question:{question}"

  messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
  ]

  qa_prompt = ChatPromptTemplate.from_messages(messages)

  llm = ChatOpenAI(model = "gpt-4",temperature = 0,openai_api_key = openai_api_key)

  memory = ConversationBufferMemory(memory_key = "chat_history",return_messages = True)

  vectorstore = load_vectorstore(
    qdrant_api_key = qdrant_api_key,
    qdrant_host = qdrant_host,
    qdrant_collection_name = coleccion,
    openai_api_key = openai_api_key
  )

  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectorstore.as_retriever(),
    memory = memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}, # noqa
    return_source_documents=False,
    verbose=False,
  )

  response = conversation_chain({
    "question": user_question,
    "chat_history": []
  })

  return response["answer"]

def main():

  # st.set_page_config(page_title = "Conversa conmigo.")
  if authenticate_user_hoy():
    st.header("Conversa conmigo y pregúntame lo que quieras!")
    st.image("robot_profesional.png", width=400)
    z = get_remain_obs(usuario=st.session_state.usuario)
    st.markdown(f"***Cuentas con {z} consultas.***")
    st.session_state.ultimas_preguntas = ultimas_preguntas(
      usuario=st.session_state.usuario,
    )

    if "password_correct" not in st.session_state:
      st.session_state.password_correct = ""

    if "usuario" not in st.session_state:
      st.session_state.usuario = ""

    if 'clicked' not in st.session_state:
      st.session_state.clicked = False

    if "psswd" not in st.session_state:
      st.session_state.psswd = ""

    if "password" not in st.session_state:
      st.session_state.password = ""

    if "ultimas_preguntas" not in st.session_state:
      st.session_state.ultimas_preguntas = ""

    if "conversation" not in st.session_state:
      st.session_state.conversation = []

    if "response" not in st.session_state:
      st.session_state.response = ""

    user_question = st.text_input("Realiza tu pregunta.")

    with st.sidebar:
      st.subheader("Tus últimas preguntas")
      st.write(st.session_state.ultimas_preguntas)
    if st.button('Pregunta'):
      if user_question and z>0:
        with st.spinner("Procesando"):
        
          st.session_state.response = get_completion(
            user_question=str(user_question),
            model = "gpt-3.5-turbo-instruct",
            usuario = st.session_state.usuario
          )
          handle_user_input(user_question)
          descontar_obs(usuario=st.session_state.usuario)
      elif z==0:
        st.write("Lo siento, no cuentas con consultas suficientes.")

if __name__ == "__main__":

  # main()
  if authenticate_user_hoy():
    st.header("Pregúntame lo que quieras acerca del libro! :books:")
    st.image("eldoc.png", width=400)
    if "preguntas_recomendadas" not in st.session_state:
        st.session_state.preguntas_recomendadas = ""

    if "conversation" not in st.session_state:
      st.session_state.conversation = []

    if "response" not in st.session_state:
      st.session_state.response = ""

    if "coleccion" not in st.session_state:
      st.session_state.coleccion = ""

    if "chat_history" not in st.session_state:
      st.session_state.chat_history = ""

    if "user" not in st.session_state:
      st.session_state.user = ""

    if "password" not in st.session_state:
      st.session_state.password = ""

    if "prompt_chatbot" not in st.session_state:
      st.session_state.prompt_chatbot = ""

    user_question = st.text_input("Realiza preguntas acerca de tus documentos.")

    if user_question:

      st.session_state.response = get_conversation_chain(str(user_question),api_key,st.session_state.coleccion,instrucciones = st.session_state.prompt_chatbot)

      handle_user_input(user_question)

    with st.sidebar:

      st.subheader("Tus documentos pdf.")
      pdf_docs = st.file_uploader(
        "Ingresa tus documentos pdf y da click en 'Procesar'",
        accept_multiple_files = True
      )

      st.subheader("También puedes elegir un documento existente")
      st.session_state.coleccion = st.selectbox(
        "También puedes elegir un documento existente",
        obtener_colecciones(
          qdrant_api_key = qdrant_api_key,
          qdrant_host = qdrant_host, 
        )
      )

      if pdf_docs is not None:
        for uploaded_file in pdf_docs:
          file_name = uploaded_file.name

      st.session_state.prompt_chatbot = st.text_input("Ingresa las indicaciones para las respuestas del bot.")

      if st.button("Procesar"):
        if pdf_docs != []:
          if st.session_state.prompt_chatbot == "":
            st.session_state.prompt_chatbot = "Eres el mejor analista de texto."

          with st.spinner("Procesando"):

            raw_text = get_pdf_text(pdf_docs)

            text_chunks = get_text_chunks(raw_text)

            st.session_state.coleccion = f"coleccion_{file_name}"

            create_vectorstore(
              text_chunks=text_chunks,
              coleccion = st.session_state.coleccion,
              qdrant_api_key=qdrant_api_key,
              qdrant_host=qdrant_host,
              openai_api_key = api_key
            )

            vectorstore = load_vectorstore(
              qdrant_api_key = qdrant_api_key,
              qdrant_host = qdrant_host,
              qdrant_collection_name = st.session_state.coleccion,
              openai_api_key = api_key
            )
            del pdf_docs

        elif st.session_state.coleccion != "":

          if st.session_state.prompt_chatbot == "":
            st.session_state.prompt_chatbot = "Eres el mejor analista de texto."
