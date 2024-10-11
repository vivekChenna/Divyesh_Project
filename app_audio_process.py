from fastapi import FastAPI
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import JinaEmbeddings
import os

load_dotenv()
app = FastAPI()
jina_api_key = os.getenv("JINA_API_KEY")

class AudioFile(BaseModel):
    file_url: str

def audio_to_text(api_url, audio_url):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "url": audio_url
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()  # Return the JSON response if successful
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def store_vector(text_chunks):
    embeddings = JinaEmbeddings(
        jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en"
    )
    vector_store = PGVector.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        connection_string="postgresql://postgres.caevgqcsjzagehkglzcf:XrEbgnNwvEK9hff5@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"        # postgresql://divyesh:123@localhost:6024/andai-chat
    )
    return vector_store


@app.post("/upload-audio-file-")
def upload_audio_file(audio_file: AudioFile):
    response = audio_to_text(api_url="https://chatbot-rsv7.onrender.com/audio-to-text/?model=whisper-large-v3&lang=en", audio_url=audio_file.file_url)
    text = response['transcription'].get('text')
    chunks = get_text_chunks(text)
    vector_store = store_vector(chunks)
    PGVector.delete_collection(self=vector_store)
    vector_store = store_vector(chunks)
    if vector_store:
        return {"message": "Audio file uploaded successfully"}

    return {"error": "An error occurred while uploading the audio file"}

