import os
import json
from datetime import datetime
import torch
from openai import OpenAI
import torch.nn.functional as F

# 默认导入 prompts_exp，但可以通过 persona_module 覆盖
from prompts_exp import *

class InvalidAPIKeyError(Exception):
    pass

class ChatBot():
    def __init__(self, config):
        self.config = config
        self.client = OpenAI()
        self.history = []
        self.refine_history = []
        self.persona_module = None  # 用于存储当前选择的 persona 模块
        # add history path and history name
        if not os.path.exists(self.config['CHATBOT']['HISTORY_PATH']):
            os.makedirs(self.config['CHATBOT']['HISTORY_PATH'], exist_ok=True)
        self.chat_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_main.json')
        self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], self.chat_name + '_refine.json')
    
    def set_persona_module(self, persona_module):
        """设置当前使用的 persona 模块"""
        self.persona_module = persona_module
    
    def get_prompt(self, prompt_name):
        """获取当前 persona 的 prompt，如果没有设置则使用默认的"""
        if self.persona_module and hasattr(self.persona_module, prompt_name):
            return getattr(self.persona_module, prompt_name)
        else:
            # 使用默认的 prompts_exp
            return globals().get(prompt_name, "")

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
        system_prompt = self.get_prompt("REFINE_WITH_PHRASE_SYSTEM_PROMPT")
        refine_prompt = self.get_prompt("REFINE_WITH_PHRASE_PROMPT")
        messages = [{"role": "system", "content": system_prompt}]
        prompt = refine_prompt.format(user_input=user_input)
        if phrase_content is not None:
            prompt += f"\n\nREFERENCE:\n{phrase_content}"
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.config['MODEL_TYPES']['QA_MODEL'],
            temperature=0.5,
            messages=messages
        )
        response = completion.choices[0].message.content
        
        messages.append({"role": "assistant", "content": response})
        self.update_refine_history({"messages": messages, "model": self.config['MODEL_TYPES']['QA_MODEL']})
        print(f"====="*10)
        print(f"Input Suggestion: {response}")
        return response


    def set_chat_type(self, chat_type="default"):
        # add system prompt
        if chat_type == "default":
            system_prompt = self.get_prompt("TEST_RAG_SYSTEM_PROMPT")
            self.update_history("system", system_prompt, self.config['MODEL_TYPES']['QA_MODEL'])
        elif chat_type == 'conversation':
            system_prompt = self.get_prompt("CONVERSATION_SYSTEM_PROMPT")
            self.update_history("system", system_prompt, self.config['MODEL_TYPES']['QA_MODEL'])


    

    def chat_intro(self, event_name, event_desc, event_obj, event_conv, user_role, ai_role, ai_starter):
        
        intro_prompt = self.get_prompt("START_INTRO")
        prompt = intro_prompt.format(
            ai_role=ai_role, 
            user_role=user_role, 
            event_name=event_name, 
            event_desc=event_desc, 
            event_obj=event_obj, 
            event_conv=event_conv, 
            ai_starter=ai_starter
        )
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL'])

        completion = self.client.chat.completions.create(
            model=self.config['MODEL_TYPES']['QA_MODEL'],
            temperature=0.5,
            messages=self.history
        )
        response = completion.choices[0].message.content
        # update assistant history
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response
    
    def chat_start_phase1(self, event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role):
        
        start_phase1_prompt = self.get_prompt("START_PHASE1")
        prompt = start_phase1_prompt.format(
            ai_role=ai_role, 
            user_role=user_role,
            event_name=event_name, 
            event_desc=event_desc, 
            event_obj=event_obj, 
            event_point=event_point, 
            event_conv=event_conv, 
            event_que=event_que
        )
        # add user input
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL']) 

        completion = self.client.chat.completions.create(
            model=self.config['MODEL_TYPES']['QA_MODEL'],
            temperature=0.5,
            messages=self.history
        )   
        response = completion.choices[0].message.content
        # update assistant history
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response


    def chat_continue_phase1(self, event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role, user_input, data_content, dict_content):
        # Format the data
        continue_phase1_prompt = self.get_prompt("CONTINUE_PHASE1")
        prompt = continue_phase1_prompt.format(
            ai_role=ai_role, 
            user_role=user_role,
            event_name=event_name, 
            event_desc=event_desc, 
            user_input=user_input, 
            event_obj=event_obj, 
            event_point=event_point, 
            event_conv=event_conv, 
            event_que=event_que
        )
        if data_content is not None:
            prompt += f"\n\nDATABASE REFERENCE:\n{data_content}"
        if dict_content is not None:
            prompt += f"\n\nDICTIONARY REFERENCE:\n{dict_content}"
        # add user input
        self.update_history("user", prompt, self.config['MODEL_TYPES']['QA_MODEL']) 
        completion = self.client.chat.completions.create(
            model=self.config['MODEL_TYPES']['QA_MODEL'],
            temperature=0.5,
            messages=self.history
        )
        response = completion.choices[0].message.content
        # update assistant history
        self.update_history("assistant", response, self.config['MODEL_TYPES']['QA_MODEL'])
        print(f"====="*10)
        print(f"Responce: {response}")
        return response