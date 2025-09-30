import os, requests, sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# Build LLM off of API key (adapted code from AgentDemo.py into a class)
class TAMULLM(LLM):
    def _call(self, prompt, stop=None):
        API_KEY = os.getenv("TAMU_API_KEY")
        API_ENDPOINT = os.getenv("TAMU_API_ENDPOINT", "https://chat-api.tamu.ai")

        if not API_KEY:
            print("Invalid API key/key wasnt set properly")
            sys.exit(1)

        headers = {"Authorization": f"Bearer {API_KEY}"}
        chat_url = f"{API_ENDPOINT}/api/chat/completions"
        body = {
            "model": "protected.llama3.2",
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = requests.post(chat_url, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            raise ValueError(f"API Error {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "tamu-llm"

# Vector basis
def build_vectorstore(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        doc_pages = loader.load()
        # Add manual name as metadata
        for d in doc_pages:
            d.metadata["manual"] = os.path.basename(path)
        docs.extend(doc_pages)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    # Local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # create FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# prompt
template = """You are a helpful lab assistant answering questions about labs.

Based on the context below, answer the question. Be specific and include relevant details from the provided documents.

If you find relevant information, provide a clear answer and cite your sources using the format (ManualName.pdf, page N).

If the information is common & could be helpful in the context of the documents, include that information.

If the information is truly not in the context, then say you don't have that information.

Context:
{context}

Question: {question}

Answer (include citations):"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Query system
def make_qa_chain(vectorstore):
    llm = TAMULLM()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

# demo
if __name__ == "__main__":
    manuals = ["214_Manuals/Lab1.pdf", "214_Manuals/Lab3.pdf"] 
    vectorstore = build_vectorstore(manuals)
    qa = make_qa_chain(vectorstore)

    query = "What are the relevant equations for Lab 3?"
    result = qa.invoke(query)

    print("\n[Answer]")
    print(result["result"])

    print("\n[Sources Used]")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['manual']} (page {doc.metadata.get('page', '?')})")
