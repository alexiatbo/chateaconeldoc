# Script para la nueva app

import ast
import hmac
import json
import openai
import os
import requests
import streamlit as st
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
qdrant_host = st.secrets["QDRANT_HOST"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

def load_json(bin_id,X_Master_Key=X_Master_Key):

  url = f'https://api.jsonbin.io/v3/b/{bin_id}'

  headers = {
    'X-Master-Key': X_Master_Key
  }

  req = requests.get(url, json=None, headers=headers).content

  data_str = req.decode('utf-8')
  try:
    data_dict = json.loads(data_str)
    record_dict = data_dict["record"]
  except:
    record_dict = ast.literal_eval(data_str)

  return record_dict

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

  output = [i for i in nombres_colecciones if '_medicina' in i]
  return output

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

def get_conversation_chain(user_question,openai_api_key,coleccion):

  instrucciones = '''Eres un experto en analizar preguntas acerca del texto 
  compartido y obtener conclusiones acertadas. Eres el asistente personal 
  de un estudiante de medicina, responde de la mejor manera asumiendo tu rol.
  No inventes nada que no sepas. 
  SOLO PUEDES RESPONDER LAS PREGUNTAS DEL USUARIO, BASANDOTE EN EL TEXTO
  QUE SE TE COMPARTA DE LOS LIBROS. DE OTRA FORMA SERÁS CASTIGADO.
  '''
  
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

    # if "prompt_chatbot" not in st.session_state:
    #   st.session_state.prompt_chatbot = ""

    user_question = st.text_input("Realiza preguntas acerca de tus documentos.")

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

      if st.button("Procesar"):
        if pdf_docs != []:

          with st.spinner("Procesando"):

            raw_text = get_pdf_text(pdf_docs)

            text_chunks = get_text_chunks(raw_text)

            st.session_state.coleccion = f"coleccion_{file_name}_medicina"

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

    if st.button('Pregunta'):
      # if user_question: z 

      st.session_state.response = get_conversation_chain(str(user_question),api_key,st.session_state.coleccion)
      # st.session_state.response = get_conversation_chain(str(user_question),api_key,st.session_state.coleccion,instrucciones = st.session_state.prompt_chatbot)

      handle_user_input(user_question)


