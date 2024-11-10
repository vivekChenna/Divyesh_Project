from fastapi import FastAPI, HTTPException
from groq import Groq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
import requests
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Define the Item model for incoming requests
class Item(BaseModel):
    image_url: str
    user_query: str

def encode_image(image_url: str) -> str:
    """
    Fetch and encode an image from a given URL to Base64.

    Args:
        image_url (str): URL of the image to fetch.

    Returns:
        str: Base64 encoded image string.

    Raises:
        HTTPException: If the image cannot be fetched.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        return base64.b64encode(response.content).decode('utf-8')
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

@app.post("/call-groq-model/")
def call_groq_model(item: Item):
    """
    Call the GROQ model with an image URL and user query.

    Args:
        item (Item): The input data containing image URL and user query.

    Returns:
        dict: Response containing the message from the model.

    Raises:
        HTTPException: If the GROQ API key is not set or if an error occurs during the API call.
    """
    api_key = os.getenv("GROQ_API_KEY")

    # Ensure the API key is set
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ API key not found.")

    # Initialize the GROQ client
    client = Groq(api_key=api_key)

    # Encode the image to Base64
    base64_image = encode_image(item.image_url)

    # Prepare the messages for the API call
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {"type": "text", "text": item.user_query},
            ],
        }
    ]

    try:
        # Call the GROQ API to get a completion
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the message content from the response
        message = completion.choices[0].message.content
        return {"message": message}

    except Exception as e:
        # Handle any exceptions that occur during the API call
        raise HTTPException(status_code=500, detail=f"API call error: {str(e)}")

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
