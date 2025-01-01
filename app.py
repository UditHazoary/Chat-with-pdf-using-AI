import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader =PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from provided context,make sure to provide all the details,if the anse is not in
    provided context - just say ,"answer is not avilable in the context",dont provide wrong answers.\n\n
    Context: \n{context}?\n
    Question : \n{question}?\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type ="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Create a placeholder for the "AI is thinking..." message
    thinking_placeholder = st.empty()
    thinking_placeholder.write("🤖 AI is thinking...")

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # Clear the "thinking" message
    thinking_placeholder.empty()    

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="wide")

    # Custom inline CSS for sidebar and main content styling
    st.markdown("""
    <style>
        /* Sidebar styling */
        .sidebar .sidebar-content {
            padding: 20px;
        }

        /* Main content styling */
        .main .block-container {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Header and text colors */
        h1, h2, h3 {
            color: #2C3E50;
        }

        /* Footer style */
        footer {
            text-align: center;
            font-size: 12px;
            padding-top: 10px;
            color: #555;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with an image
    st.image("header_image2.jpg", use_container_width=True)

    # Header
    st.header("📄 Chat with PDF Files Using AI")
    st.subheader("Upload, Process, and Ask Questions Effortlessly!")

    # Sidebar
    with st.sidebar:
        st.image("side_bar_image.jpg", use_container_width=True)
        st.title("📁 File Processing")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("📥 Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete! 🎉")

    # User question
    user_question = st.text_input("Type your question below:")
    if user_question:
        user_input(user_question)

    # Footer
    st.markdown("""
        ---
        <footer>
            <h5>Built with ❤️ using Streamlit and LangChain</h5>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
