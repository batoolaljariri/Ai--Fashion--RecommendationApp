import os
import base64
import json
import re
from PIL import Image
from io import BytesIO
from groq import Groq
from pydantic import BaseModel
from typing import List
 
 
# Class holding schema
class FashionBaseModel(BaseModel):
    clothing_category: str
    color: str
    style: str
    occasion: str
    suitable_for_weather: str
    description: str
 
 
# Class with a list of objects
class FashionModel(BaseModel):
    clothes: List[FashionBaseModel]
 
 
def initialize_client(api_key: str) -> Groq:
    """Initialize the Groq client."""
    return Groq(api_key=api_key)
 
 
def process_image(client, image_path: str, schema: dict):
    """Process a single image and return its features."""
    try:
        # Convert image to base64
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
 
        # Prepare prompt request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an expert in fashion and clothes. Feature extract clothes within images.
                         Return JSON output only if it validates against the provided schema.
                          Do not generate schemas, or anything other than a JSON array of clothing features. The JSON object must use the schema:
                           {json.dumps(schema, indent=2)}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
 
        # Make API call
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            stream=False,
            stop=None,
            response_format={"type": "json_object"},
        )
 
        # Extract response
        response = completion.choices[0].message.content
        return extract_json_from_response(response)
 
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
 
 
def extract_json_from_response(response: str):
    """Extract the JSON object from the API response."""
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
    return None
 
 
def main():
    # Initialize Groq client
    client = initialize_client(api_key="APIKEY")
 
    # Prompt the user for an image file
    image_path = input("Enter the path to your image file: ").strip()
    if not os.path.exists(image_path):
        print("The specified file does not exist. Please check the path and try again.")
        return
 
    # Define the schema
    schema = FashionModel.model_json_schema()
 
    # Process the image
    features = process_image(client, image_path, schema)
    if features:
        print("\nExtracted Features:")
        for key, value in features.items():
            print(f"{key}: {value}")
    else:
        print("Failed to extract features from the image.")
 
 
# if __name__ == "__main__":
#     main()
