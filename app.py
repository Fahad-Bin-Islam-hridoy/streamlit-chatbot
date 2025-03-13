import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import base64

# Function to convert image to Base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get base64 string of the corona logo
image_base64 = get_image_base64("corona_image.png")

# 🔹 **Initialize OpenAI API Key**

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the key securely



# 🔹 **Custom CSS for Full Centering, Better Spacing & Button Visibility**
st.markdown(
    """
    <style>
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background-color: white;
            color: #27529f;
        }
        .title-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .logo {
            width: 150px;
            margin-bottom: 15px;
        }
        .button-container {
            margin-top: 10px;
        }
        .conversation-box {
            max-width: 80%;
            margin: auto;
            text-align: left;
            padding: 10px;
        }
        .stButton>button {
            background-color: #27529f;
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            text-align: center;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{image_base64}" width="150">
    </div>
    """,
    unsafe_allow_html=True
)
# 🔹 **Header & Logo Centered**
st.markdown(
    """
    <div class="title-container">   
        <h3>📢 Product Review Chatbot</h3>
        <h5>Ask about product reviews, durability, and performance!</h5>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔹 **Load & Process Review Data**
@st.cache_data
def load_data():
    df = pd.read_excel("Toilets_Reviews_Processed.xlsx")  # Ensure the file exists
    df = df[["Product Name", "Stars", "Translated Review"]]  # Select relevant columns
    df["content"] = df.apply(lambda row: f"Product: {row['Product Name']}\nRating: {row['Stars']}\nReview: {row['Translated Review']}", axis=1)
    return df

df = load_data()

# 🔹 **Chunking for Retrieval**
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
docs = text_splitter.split_text("\n".join(df["content"].tolist()))

# 🔹 **Create or Load Vector DB**
@st.cache_resource
def create_or_load_vector_db():
    if os.path.exists("review_faiss_index_2"):
        return FAISS.load_local("review_faiss_index_2", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_texts(docs, embedding=embeddings)
        vector_db.save_local("review_faiss_index_@")
        return vector_db

vector_db = create_or_load_vector_db()

# 🔹 **Initialize LLM Model**
llm = ChatOpenAI(model_name="gpt-4o")

# 🔹 **Create Retrieval-based QA system**
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

# 🌟 **User Input Box (Centered)**
st.markdown("### 🔍 Ask your question:")
query = st.text_input("", key="query")

# Initialize conversation history if not present
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize response to avoid NameError
response = ""

# 🔹 **Answer Button**
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Get Answer"):
    if query:
        response = qa_chain.run(query)  # ✅ Fetch the actual chatbot response
        st.markdown("### 💬 Chatbot Response:")
        st.write(response)

        # Append user query and chatbot response to history
        st.session_state.history.append(("You", query))
        st.session_state.history.append(("Chatbot", response))
    else:
        st.warning("⚠️ Please enter a question!")
st.markdown("</div>", unsafe_allow_html=True)

# 🔹 **Display Conversation History**
st.markdown("### 💬 Conversation History:")
st.markdown('<div class="conversation-box">', unsafe_allow_html=True)

for user, text in st.session_state.history:
    st.markdown(f"**{user}:** {text}")

st.markdown("</div>", unsafe_allow_html=True)
