import os
import json
from datetime import datetime
import torch
import random
from openai import OpenAI
import torch.nn.functional as F


class InContextLearner():
    def __init__(self, config):
        self.config = config
        self.example_path = self.config['IN_CONTEXT']['EXAMPLE_PATH']
        self.examples = json.load(open(self.example_path))
        self.conversation_with_types = {}
        for item in self.examples:
            if item["event"] in self.conversation_with_types:
                self.conversation_with_types[item["event"]].append(item)
            else:
                self.conversation_with_types[item["event"]] = [item, ]

    def get_types(self):
        event_types = list(self.conversation_with_types.keys())
        event_descs = [self.conversation_with_types[name][0]["description"] for name in event_types]
        return event_types, event_descs
    
    def get_roles(self, event_type):
        ai_role = 'assistant'
        user_role = 'User'
        for round in self.conversation_with_types[event_type][0]["conversation"]:
            if "users" in round:
                user_role = round["users"]
            elif "control" in round:
                ai_role = round["control"]
        return ai_role, user_role
    
    def get_incontext_examples(self, event_type):
        candidates = self.conversation_with_types[event_type]
        num_examples = min(self.config['IN_CONTEXT']['NUM_SAMPLES'], len(candidates))
        selected_samples = random.sample(candidates, num_examples)
        print(f"====="*10)
        print(f"Select {len(selected_samples)} examples from event: {event_type}")
        is_user_start = "users" in selected_samples[0]["conversation"][0]
        print(f"Is this event start by user: {is_user_start}")
        if not is_user_start:
            assert "control" in selected_samples[0]["conversation"][0], f"Error Starter: {str(selected_samples[0]["conversation"][0])}"
            ai_starter = selected_samples[0]["conversation"][0]["utterance"]
        else:
            ai_starter = None
        return selected_samples, is_user_start, ai_starter
