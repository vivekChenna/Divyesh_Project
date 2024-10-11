from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import JinaEmbeddings
import os

load_dotenv()
app = FastAPI()
jina_api_key = os.getenv("JINA_API_KEY")
API = os.getenv("GROQ_API_KEY")

def get_vector_store():
    try:
        embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v2-base-en"
        )
        vector_store = PGVector(
            connection_string="postgresql://postgres.caevgqcsjzagehkglzcf:XrEbgnNwvEK9hff5@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres",
            embedding_function=embeddings,
            use_jsonb=True,
        )
        return vector_store
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_conversational_chain(vector_store):
    llm = ChatGroq(temperature=0.1, groq_api_key=API, model_name="llama-3.1-70b-versatile")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain

class UserQuestion(BaseModel):
    user_question: str

@app.post("/chat-with-audio")
def chat_with_audio(user: UserQuestion):
    vector_store = get_vector_store()
    if vector_store:
        conversation_chain = get_conversational_chain(vector_store)
        response = conversation_chain.invoke({'question': user.user_question})
        return response
    return {"error": "An error occurred while searching the vector store"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
