import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()
st.set_page_config(initial_sidebar_state="collapsed")

# Load the environment variables from either Streamlit Secrets or .env file
LOCAL_SECRETS = False

if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_ENDPOINT = st.secrets["ASTRA_DB_ENDPOINT"]
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "brooksfield"

os.environ["LANGCHAIN_PROJECT"] = "blueillusion"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = DataAPIClient(st.secrets["ASTRA_DB_APPLICATION_TOKEN"])
database = client.get_database(st.secrets["ASTRA_DB_ENDPOINT"])
collection = database.get_collection(ASTRA_DB_COLLECTION)  # Use correct method to access collection

import re

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌", unsafe_allow_html=True)

    def on_llm_end_of_transmission(self):
        self.container.markdown(self.text, unsafe_allow_html=True)

top_k_vectorstore = 8
top_k_memory = 3

global embedding
global vectorstore
global retriever
global model
global chat_history
global memory

def check_username():
    greeting_message = "Welcome to the contract analysis assistant. How can I assist you?"
    username_prompt = "Please enter your username to continue:"

    if 'username_valid' not in st.session_state:
        st.session_state.username_valid = False  # Ensure it's explicitly set for new sessions
        st.write(greeting_message)
    
    username = st.text_input(username_prompt, key='username')
    
    if username:
        if not st.session_state.get('username_valid', False):  # Check if it's the first time the username is set
            st.session_state.username_valid = True
            st.session_state.user = username
            #initial_greeting()  # Call greeting after setting the username
    else:
        st.session_state.username_valid = False

    return st.session_state.username_valid

def initial_greeting():
    if "initial_greeted" not in st.session_state:
        st.session_state.initial_greeted = True
        st.session_state.messages = [  # Initialize and set messages directly here
            AIMessage(content="Hello! I'm here to help you with your contract questions. You can ask me things like:"),
            AIMessage(content="[1. What is this document about?](#question1)"),
            AIMessage(content="[2. What are the terms and conditions of the contract?](#question2)"),
            AIMessage(content="[3. Can you explain specific clauses mentioned in the document?](#question3)")
        ]
        for message in st.session_state.messages:
            st.chat_message(message.type).markdown(message.content)  # Display messages in chat

# Ensure initialization at the start of your script

if "messages" not in st.session_state:
   st.session_state.messages = []

def logout():
    keys_to_delete = ['username_valid', 'user', 'messages', 'selected_filename']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    if 'load_chat_history' in globals():
        load_chat_history.clear()
    if 'load_memory' in globals():
        load_memory.clear()
    if 'load_retriever' in globals():
        load_retriever.clear()

if not check_username():
    st.stop()

username = st.session_state.user

@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    return AstraDBVectorStore(
        embedding=embedding,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_ENDPOINT,
    )

@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="openai.gpt-3.5"):
    return ChatOpenAI(
        temperature=0.2,
        model='gpt-3.5-turbo',
        streaming=True,
        verbose=True
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history(username):
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

@st.cache_data()
def load_prompt():
    template = """You're an expert assistant in analyzing and interpreting contracts. Your goal is to help users understand the details, clauses, and implications of their contracts. Provide clear and precise answers based on the context and previous chat history.
If the user mentions a specific document or contract, use the relevant information from that document to answer their question.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in English"""

    return ChatPromptTemplate.from_messages([("system", template)])

if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_greeting()

embedding = load_embedding()
vectorstore = load_vectorstore()
retriever = load_retriever()
chat_history = load_chat_history(username)
memory = load_memory()
prompt = load_prompt()

model_id = 'openai.gpt-3.5'
model = load_model(model_id)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def embed_and_store_text(text, filename):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"filename": filename}) for chunk in texts]
    vectorstore.add_documents(documents)

# Fetch all filenames
def get_all_filenames():
    all_documents = collection.find({}, projection={"metadata.filename": True})
    filenames = list(set(doc['metadata']['filename'] for doc in all_documents if 'metadata' in doc and 'filename' in doc['metadata']))
    return filenames

if st.session_state.messages:
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)

# Sidebar with document upload and selection
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        embed_and_store_text(text, uploaded_file.name)
        st.success("PDF content has been embedded and stored successfully.")
    
    filenames = get_all_filenames()
    filenames.insert(0, "ALL")
    selected_filename = st.selectbox("Select a document for search:", filenames, key='selected_filename')

# Function to handle clickable questions
def handle_click(question):
    st.session_state['chat_input'] = question

if question := st.chat_input("How can I help you?", key='chat_input'):
    st.session_state.messages.append(HumanMessage(content=question))

    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        if "last_selected_filename" not in st.session_state or st.session_state.last_selected_filename != selected_filename:
            st.session_state.last_selected_filename = selected_filename
            history = {}
        else:
            history = memory.load_memory_variables({})

        print(f"Using memory: {history}")

        def get_filtered_documents(query):
            query_vector = embedding.embed_query(query)
            if selected_filename == "ALL":
                documents = collection.find(
                    {},
                    sort={"$vector": query_vector},
                    limit=10,
                    projection={"content": True},
                )
            else:
                documents = collection.find(
                    {"metadata.filename": selected_filename},
                    sort={"$vector": query_vector},
                    limit=10,
                    projection={"content": True},
                )
            return [doc['content'] for doc in documents]

        inputs = RunnableMap({
            'context': lambda x: "\n\n".join(get_filtered_documents(x['question'])),
            'chat_history': lambda x: x['chat_history'] if "last_selected_filename" in st.session_state and st.session_state.last_selected_filename == selected_filename else '',
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        print(f"Response: {response}")
        
        content = response.content

        # Write the final answer without the cursor
        response_placeholder.markdown(content)

        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))

# Add the greeting messages with clickable links
if "initial_greeted" not in st.session_state:
    #initial_greeting()
    st.session_state.initial_greeted = True
    
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content, unsafe_allow_html=True)