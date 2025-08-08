# Imports
from pdfminer.high_level import extract_text
import base64
import io
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd 
import json
import torch
import numpy as np
from rich import print
import time


def clean_contents(raw_doc, overlap=10, text_length=100):
    filename = raw_doc['filename']
    text = raw_doc['text']
    description = raw_doc['pages_description']

    def clean_str(content):
        content = content.replace('\f', '')
        content = content.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
        content = re.sub(r"(?<=\n)\d{1,2}", "", content)
        content = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", content, flags=re.IGNORECASE)
        return content
    
    clean_text = clean_str(text)
    clean_description = [clean_str(desc) for desc in description]

    clean_contents = []
    # add description
    for i, content in enumerate(clean_description):
        clean_contents.append({'meta': f"File: <<{filename}>> Desc Page-{i}",
                               "content": content})
    
    # add raw text with length 1000 and overlap 100
    text_list = clean_text.split(" ")
    text_id = 0
    assert text_length > overlap
    while(len(text_list) > text_length):
        clean_contents.append({'meta': f"File: <<{filename}>> Text Index-{text_id}",
                               "content": " ".join(text_list[:text_length])})
        text_list = text_list[text_length-overlap:]
        text_id += 1
    clean_contents.append({'meta': f"File: <<{filename}>> Text Index-{text_id}",
                               "content": " ".join(text_list)})
    return clean_contents



def get_contents_with_embedding(raw_doc, overlap=10, text_length=100, model_type="text-embedding-3-large"):
    contents = clean_contents(raw_doc, overlap=overlap, text_length=text_length)

    client = OpenAI()
    def get_embeddings(text):
        embeddings = client.embeddings.create(
        model=model_type,
        input=text,
        encoding_format="float"
        )
        return embeddings.data[0].embedding
    
    embeddings = []
    with tqdm(total=len(contents)-1) as pbar:
        for item in contents:
            # set sleep to avoid trigger tokens per minute (TPM) limit
            time.sleep(1) 
            embeddings.append(get_embeddings(item['content']))
            pbar.update(1)
    embeddings = torch.FloatTensor(embeddings)

    contents_with_embed = {'meta': [item['meta'] for item in contents],
                           'content': [item['content'] for item in contents],
                           'embedding': embeddings}
    
    return contents_with_embed


def get_dictionary_with_embedding(raw_dict, dict_name, model_type="text-embedding-3-large"):
    contents = [{'meta': f"File: <<{dict_name}>> Dict Index-{index}", "content": {str(item)}} for index, item in enumerate(raw_dict)]

    client = OpenAI()
    def get_embeddings(text):
        embeddings = client.embeddings.create(
        model=model_type,
        input=text,
        encoding_format="float"
        )
        return embeddings.data[0].embedding
    
    embeddings = []
    with tqdm(total=len(contents)-1) as pbar:
        for item in contents:
            # set sleep to avoid trigger tokens per minute (TPM) limit
            embeddings.append(get_embeddings(item['content']))
            pbar.update(1)
    embeddings = torch.FloatTensor(embeddings)

    contents_with_embed = {'meta': [item['meta'] for item in contents],
                           'content': [item['content'] for item in contents],
                           'embedding': embeddings}
    
    return contents_with_embed

    
     


