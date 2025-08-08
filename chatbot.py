import os
import json
from datetime import datetime
import torch
from openai import OpenAI
import torch.nn.functional as F

from prompts import *

class InvalidAPIKeyError(Exception):
    pass

class ChatBot():
    def __init__(self, config):
        self.config = config
        self.client = OpenAI()
        self.history = []
        self.refine_history = []
        # add history path and history name
        if not os.path.exists(self.config['CHATBOT']['HISTORY_PATH']):
            os.makedirs(self.config['CHATBOT']['HISTORY_PATH'], exist_ok=True)
        self.chat_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_main.json')
        self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_refine.json')
        

    def clear_history(self):
        print(f"==> Delete {len(self.history)} items in history")
        self.history = []
        self.refine_history = []
        print(f"==> Update experiment id")
        self.chat_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_main.json')
        self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_refine.json')

    def display_history(self):
        for i, round in enumerate(self.history):
            print(f"[Round {i}] ({round["model"]}) {round["role"]}: {round["content"]}")
        
        
    def update_history(self, role, prompt, model_type):
        self.history.append({"role": role, "content": prompt, "model": model_type})
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)


    def update_refine_history(self, refine_chat_item):
        self.refine_history.append(refine_chat_item)
        with open(self.refine_history_path, "w", encoding="utf-8") as f:
            json.dump(self.refine_history, f, ensure_ascii=False, indent=4)


    def refine_user_input_with_phrase(self, phrase_content, user_input):
        messages = [{"role": "system", "content": REFINE_WITH_PHRASE_SYSTEM_PROMPT}]
        prompt = REFINE_WITH_PHRASE_PROMPT.format(user_input=user_input)
        if phrase_content is not None:
            prompt += f"\n\nREFERENCE:\n{phrase_content}"
        messages.append({"role": "user", "content": prompt})
        try:
            completion = self.client.chat.completions.create(
                model=self.config['MODEL_TYPES']['QA_MODEL'],
                temperature=0.5,
                messages=messages
            )
            response = completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'http_status') and e.http_status == 401:
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            raise
        messages.append({"role": "assistant", "content": response})
        self.update_refine_history({"messages": messages, "model": self.config['MODEL_TYPES']['QA_MODEL']})
        print(f"====="*10)
        print(f"Input Suggestion: {response}")
        return response


    def set_chat_type(self, chat_type="default"):
        # add system prompt
        if chat_type == "default":
            self.update_history("system", TEST_RAG_SYSTEM_PROMPT, self.config['MODEL_TYPES']['QA_MODEL'])
        elif chat_type == 'conversation':
            self.update_history("system", CONVERSATION_SYSTEM_PROMPT, self.config['MODEL_TYPES']['QA_MODEL'])

    def test(self, user_input, data_content, dict_content):
        prompt = f"INPUT PROMPT:\n{user_input}"
        if data_content is not None:
            prompt += f"\n-------\nDATABASE REFERENCE:\n{data_content}"
        if dict_content is not None:
            prompt += f"\n-------\nDICTIONARY REFERENCE:\n{dict_content}"
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL'])
        try:
            completion = self.client.chat.completions.create(
                model=self.config['MODEL_TYPES']['QA_MODEL'],
                temperature=0.5,
                messages=self.history
            )
            response = completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'http_status') and e.http_status == 401:
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            raise
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response
    

    def chat_start_response(self, event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content):
        prompt = STARTER_RESPONSE.format(ai_role=ai_role, user_role=user_role,
                                        event_name=event_name, event_desc=event_desc, user_input=user_input)
        if data_content is not None:
            prompt += f"\n\nDATABASE REFERENCE:\n{data_content}"
        if dict_content is not None:
            prompt += f"\n\nDICTIONARY REFERENCE:\n{dict_content}"
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL'])
        try:
            completion = self.client.chat.completions.create(
                model=self.config['MODEL_TYPES']['QA_MODEL'],
                temperature=0.5,
                messages=self.history
            )
            response = completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'http_status') and e.http_status == 401:
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            raise
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response
    
    def chat_start_conversation(self, event_name, event_desc, user_role, ai_role, ai_starter):
        prompt = START_CONVERSATION.format(ai_role=ai_role, user_role=user_role,
                                        event_name=event_name, event_desc=event_desc, ai_starter=ai_starter)
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL'])
        try:
            completion = self.client.chat.completions.create(
                model=self.config['MODEL_TYPES']['QA_MODEL'],
                temperature=0.5,
                messages=self.history
            )
            response = completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'http_status') and e.http_status == 401:
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            raise
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response
    
    def chat_continue_response(self, event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content):
        prompt = CONTINUE_RESPONSE.format(ai_role=ai_role, user_role=user_role,
                                        event_name=event_name, event_desc=event_desc, user_input=user_input)
        if data_content is not None:
            prompt += f"\n\nDATABASE REFERENCE:\n{data_content}"
        if dict_content is not None:
            prompt += f"\n\nDICTIONARY REFERENCE:\n{dict_content}"
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL'])
        try:
            completion = self.client.chat.completions.create(
                model=self.config['MODEL_TYPES']['QA_MODEL'],
                temperature=0.5,
                messages=self.history
            )
            response = completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'http_status') and e.http_status == 401:
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
                raise InvalidAPIKeyError("Invalid OpenAI API key.")
            raise
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response