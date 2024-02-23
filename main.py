#AIzaSyBMPDE8HmFvRHwnw5AcIvuOUGgoGaJE-cY
import pathlib
import textwrap
import spacy

from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import google.generativeai as genai

#import torch

import base64
import requests
import os
#import imgur_uploader
import cloudinary
import cloudinary.uploader

##a = input("Enter 1 to generate text, 2 to identify image: ")
a = 0

cloudinary.config(
  cloud_name = "dndb4uvex",
  api_key = "833635992464733",
  api_secret = "ApdRUu09Rb688lsQk2Ku6r7kCX4"
)

genai.configure(api_key="AIzaSyD6BnbnkQ4v8QFEpbtml5_3sMn_qeWonhQ")

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

model_text = genai.GenerativeModel('gemini-pro')
model_image = genai.GenerativeModel('gemini-pro-vision')
model_code = genai.GenerativeModel('gemini-1.0-pro')
model_id = "stabilityai/stable-diffusion-2"
#generator = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

def detect_intent(prompt):
    # Process the input prompt
    doc = nlp(prompt)
    print(doc)
    
    # Check if the prompt contains keywords related to image description
    if "describe" in prompt.lower() or "explain" in prompt.lower() and ("image" in prompt.lower() or "picture" in prompt.lower()) and ("http" in prompt.lower() or "www" in prompt.lower()) or ("https" in prompt.lower() or "www" in prompt.lower()):
        # Extract the URL from the prompt
        url_start_index = prompt.lower().find("http") if "http" in prompt.lower() else prompt.lower().find("www")
        url_end_index = prompt.lower().find(" ", url_start_index) if url_start_index != -1 else len(prompt)
        url = prompt[url_start_index:url_end_index]
        return 1 , url
    # Check if the prompt asks to create an image
    elif "create" in prompt.lower() and ("image" in prompt.lower() or "picture" in prompt.lower()):
        return 2, "hi"
    # Check if the prompt asks to create code
    elif "create" in prompt.lower() or "generate" in prompt.lower() and ("code" in prompt.lower() or "coding" in prompt.lower()):
        return 4, "hi"
    # Check if the prompt asks to summarise video ERROR
    #elif "summarise" in prompt.lower() and ("video" in prompt.lower() and ("http" in prompt.lower() or "www" in prompt.lower()) or ("https" in prompt.lower() or "www" in prompt.lower())):
     #   return 3, "hi"
    #solve this error by checking       
    else:
        return 3, "hi"  # Normal chatting

y = 1

while y == 1:

    z = input("YOU: ")

    prompt = z

    intent = detect_intent(prompt)
    print({intent[0]})
    value = {intent[0]}

    if value == {1}: 
        a = 2
    elif value == {2}:
        a = 3
    elif value == {3}:
        a = 1
    elif value == {4}:
        a = 4
    else:
        print("Error")



    if a == 1:
        if z == "exit":
            y = 2
        else:
            y = 1
            response = model_text.generate_content(z)
            parts = response.candidates[0].content.parts
            for part in parts:
                print(part.text)
        

    elif a == 2:
        if z == "exit":
            y = 2
        else:
            y = 1
            b = intent[1] + "g"
            print(b)
            urllib.request.urlretrieve( b, "1.png")
            img = Image.open("1.png")
            response = model_image.generate_content(img)
            parts = response.candidates[0].content.parts
            for part in parts:
                print(part.text)
            img.show()

    elif a == 4:
        if z == "exit":
            y = 2
        else:
            y = 1
            response = model_code.generate_content(z)
            parts = response.candidates[0].content.parts
            for part in parts:
                print(part.text)
        

    elif a == 3:
        if z == "exit":
            y = 2
        else:
            y = 1
            prompt_image_generation = input("Enter an image prompt: ")
            image_generation_prompt = prompt_image_generation
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

            body = {
            "steps": 40,
            "width": 1024,
            "height": 1024,
            "seed": 0,
            "cfg_scale": 5,
            "samples": 1,
            "text_prompts": [
                {
                "text": image_generation_prompt,
                "weight": 1
                },
                {
                "text": "blurry, bad",
                "weight": -1
                }
            ],
            }

            headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "sk-z5iOgOHQKwU2XfuQ034vEDdgJ7PJGiD4zYNE0fOC3C8NyNq7",
            }

            response = requests.post(
            url,
            headers=headers,
            json=body,
            )

            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))

            data = response.json()

            # make sure the out directory exists
            out_dir = "./out"
            if not os.path.exists("./out"):
                os.makedirs(out_dir)

            for i, image in enumerate(data["artifacts"]):
                file_path = os.path.join("/Users/apple/Desktop/CodingSpace/out", f'txt2img_{image["seed"]}.png')
                with open(f'/Users/apple/Desktop/CodingSpace/out/txt2img_{image["seed"]}.png', "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                # Path to the image file you want to upload
            image_path = file_path

                # Upload the image
            uploaded_image = cloudinary.uploader.upload(image_path)

                # Retrieve the link to the uploaded image
            image_link = uploaded_image['url']

            print("Image uploaded successfully!")
            print("Image link:", image_link)
