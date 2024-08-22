import os
import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
st.set_page_config(initial_sidebar_state="collapsed")

LOCAL_SECRETS = False

if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "brooksfield"

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(st.secrets["ASTRA_DB_ENDPOINT"])
collection = database.get_collection(ASTRA_DB_COLLECTION)

# Initialize session_state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

def check_username():
    greeting_message = "Welcome to the contract analysis assistant. How can I assist you?"
    username_prompt = "Please enter your username to continue:"

    if 'username_valid' not in st.session_state:
        st.session_state.username_valid = False
        st.write(greeting_message)
    
    username = st.text_input(username_prompt, key='username')
    
    if username:
        if not st.session_state.get('username_valid', False):
            st.session_state.username_valid = True
            st.session_state.user = username
    else:
        st.session_state.username_valid = False

    return st.session_state.username_valid

if not check_username():
    st.stop()

username = st.session_state.user

@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    return AstraDBVectorStore(
        embedding=load_embedding(),
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=st.secrets["ASTRA_DB_ENDPOINT"],
    )

@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model():
    return OpenAI(openai_api_key=OPENAI_API_KEY)

embedding = load_embedding()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
llm = load_model()

# Define the prompt template
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 
    
    context: {context}
    Question: "{question}"
    Answer:
    """
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def embed_and_store_text(text, filename):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"filename": filename}) for chunk in texts]
    vectorstore.add_documents(documents)

@st.cache_data()
def get_all_filenames():
    all_documents = collection.find({}, projection={"metadata.filename": True})
    filenames = list(set(doc['metadata']['filename'] for doc in all_documents if 'metadata' in doc and 'filename' in doc['metadata']))
    return filenames

def handle_chat(question, selected_filename):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

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

        context = "\n\n".join(get_filtered_documents(question))

        # Build the chain with prompt and LLM
        chain = LLMChain(
            llm=llm,
            prompt=ANSWER_PROMPT,
        )

        # Execute the chain
        ans = chain({"context": context, "question": question})

        # Ensure the response is a string
        response_content = str(ans["text"])

        # Display and store the response
        response_placeholder.markdown(response_content)
        st.session_state.messages.append(AIMessage(content=response_content))

if st.session_state.messages:
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        embed_and_store_text(text, uploaded_file.name)
        st.success("PDF content has been embedded and stored successfully.")
    
    filenames = get_all_filenames()
    filenames.insert(0, "ALL")
    selected_filename = st.selectbox("Select a document for search:", filenames, key='selected_filename')

if question := st.chat_input("How can I help you?", key='chat_input'):
    st.session_state.messages.append(HumanMessage(content=question))
    handle_chat(question, selected_filename)

if "initial_greeted" not in st.session_state:
    st.session_state.initial_greeted = True
    
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content, unsafe_allow_html=True)