# Imports
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
import io
import os
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd 
import json
import numpy as np
from rich import print
import time

from prompts import IMAGE_ANALYZE_PROMPT

def analyze_doc_image(img, model_type="gpt-4o"):
    # turn image to base64 data
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)
    base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')
    data_uri = f"data:image/png;base64,{base64_png}"

    # use openai to analyze image
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": IMAGE_ANALYZE_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{data_uri}"
                        }
                    }
                ]
                },
        ],
        max_tokens=1000,
        temperature=0,
        top_p=0.1
    )
    data = response.choices[0].message.content
    return data


def pdf_loader(file_name, file_path, model_type):    
    doc = {"filename": file_name}
    text = extract_text(file_path)
    doc['text'] = text

    imgs = convert_from_path(file_path)
    pages_description = []
    print(f"Analyzing pages for doc {file_name}")
    
    with tqdm(total=len(imgs)-1) as pbar:
        for img in imgs:
            # set sleep to avoid trigger tokens per minute (TPM) limit
            time.sleep(1) 
            res = analyze_doc_image(img, model_type)
            pages_description.append(res)
            pbar.update(1)
    
    doc['pages_description'] = pages_description
    return doc

#def pdf_page_loader(file_path):
#    with pdfplumber.open(file_path) as pdf:
#        pages = []
#        for page in pdf.pages:
#            pages.append(page.extract_text())
#        return pages