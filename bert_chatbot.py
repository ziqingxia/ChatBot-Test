import os
import json
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional

class BertChatBot:
    def __init__(self, config):
        self.config = config
        self.history = []
        self.refine_history = []
        
        # Initialize BERT model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except Exception as e:
            raise Exception(f"Failed to initialize BERT model: {e}")
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # History management
        if not os.path.exists(self.config['CHATBOT']['HISTORY_PATH']):
            os.makedirs(self.config['CHATBOT']['HISTORY_PATH'], exist_ok=True)
        self.chat_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_bert_main.json")
        self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_bert_refine.json")
    
    def _load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base from database files"""
        knowledge_base = []
        
        # Load from database directories
        database_path = self.config['DATABASE']['ROOT_PATH']
        if os.path.exists(database_path):
            for folder in os.listdir(database_path):
                folder_path = os.path.join(database_path, folder)
                if os.path.isdir(folder_path):
                    json_file = os.path.join(folder_path, "contents_without_embed.json")
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    knowledge_base.extend(data)
                                else:
                                    knowledge_base.append(data)
                        except Exception as e:
                            print(f"Error loading {json_file}: {e}")
        
        # Load from dictionary directories
        dictionary_path = self.config['DICTIONARY']['ROOT_PATH']
        if os.path.exists(dictionary_path):
            for folder in os.listdir(dictionary_path):
                folder_path = os.path.join(dictionary_path, folder)
                if os.path.isdir(folder_path):
                    json_file = os.path.join(folder_path, "raw_dict.json")
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    knowledge_base.extend(data)
                                else:
                                    knowledge_base.append(data)
                        except Exception as e:
                            print(f"Error loading {json_file}: {e}")
        
        return knowledge_base
    
    def clear_history(self):
        """Clear conversation history"""
        print(f"==> Delete {len(self.history)} items in history")
        self.history = []
        self.refine_history = []
        print(f"==> Update experiment id")
        self.chat_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_bert_main.json")
        self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_bert_refine.json")
    
    def update_history(self, role: str, prompt: str, model_type: str = "BERT"):
        """Update conversation history"""
        self.history.append({"role": role, "content": prompt, "model": model_type})
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
    
    def _get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Get relevant context from knowledge base using BERT embeddings"""
        if not self.knowledge_base:
            return ""
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Encode knowledge base entries
        knowledge_texts = []
        for item in self.knowledge_base:
            if isinstance(item, dict):
                # Extract text content from various possible fields
                text = ""
                for key in ['content', 'text', 'message', 'response', 'description']:
                    if key in item and item[key]:
                        text += str(item[key]) + " "
                if text.strip():
                    knowledge_texts.append(text.strip())
            elif isinstance(item, str):
                knowledge_texts.append(item)
        
        if not knowledge_texts:
            return ""
        
        # Encode knowledge texts
        knowledge_embeddings = self.model.encode(knowledge_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build context from top matches
        context_parts = []
        for idx in top_indices:
            if similarities[idx] > self.config['SEARCH']['THRESHOLD']:
                context_parts.append(knowledge_texts[idx])
        
        return "\n".join(context_parts)
    
    def _generate_response(self, user_input: str, context: str = "") -> str:
        """Generate response using BERT-based approach"""
        # This is a simplified response generation
        # In a real implementation, you might use a fine-tuned BERT model for text generation
        # or combine BERT embeddings with a language model
        
        # For now, we'll use a rule-based approach with context matching
        if context:
            # Use context to generate more informed response
            return self._generate_contextual_response(user_input, context)
        else:
            # Generate basic response based on input patterns
            return self._generate_basic_response(user_input)
    
    def _generate_contextual_response(self, user_input: str, context: str) -> str:
        """Generate response using context"""
        # Simple pattern matching with context
        user_lower = user_input.lower()
        
        # Check for common patterns in railway communication
        if "radio check" in user_lower:
            return "Radio Check OK, over."
        elif "over" in user_lower and "send" not in user_lower:
            return "Roger, over."
        elif "out" in user_lower:
            return "Roger, out."
        elif "emergency" in user_lower:
            return "Emergency acknowledged. Please provide details, over."
        elif "toa" in user_lower and "surrender" in user_lower:
            return "TOA surrender acknowledged. Please proceed with Line Clear message, over."
        elif "line clear" in user_lower:
            return "Line Clear Message Acknowledged. Please confirm all clear, over."
        else:
            # Use context to find relevant response
            context_lower = context.lower()
            if "radio" in context_lower:
                return "Radio communication protocol followed, over."
            elif "emergency" in context_lower:
                return "Emergency protocol activated, over."
            else:
                return "Message received, over."
    
    def _generate_basic_response(self, user_input: str) -> str:
        """Generate basic response without context"""
        user_lower = user_input.lower()
        
        # Basic pattern matching
        if "over" in user_lower:
            return "Roger, over."
        elif "out" in user_lower:
            return "Roger, out."
        elif "emergency" in user_lower:
            return "Emergency acknowledged, over."
        elif "radio check" in user_lower:
            return "Radio Check OK, over."
        else:
            return "Message received, over."
    
    def test(self, user_input: str, data_content: Optional[str] = None, dict_content: Optional[str] = None) -> str:
        """Main test method - equivalent to OpenAI chatbot's test method"""
        # Get relevant context
        context = self._get_relevant_context(user_input)
        
        # Add data_content and dict_content if provided
        if data_content:
            context += f"\n\nDATABASE REFERENCE:\n{data_content}"
        if dict_content:
            context += f"\n\nDICTIONARY REFERENCE:\n{dict_content}"
        
        # Update history
        prompt = f"INPUT PROMPT:\n{user_input}"
        if context:
            prompt += f"\n-------\nCONTEXT:\n{context}"
        
        self.update_history("user", prompt, "BERT")
        
        # Generate response
        response = self._generate_response(user_input, context)
        
        # Update history with response
        self.update_history("assistant", response, "BERT")
        
        print(f"====="*10)
        print(f"BERT Response: {response}")
        return response
    
    def chat_start_response(self, event_name: str, event_desc: str, user_role: str, 
                           ai_role: str, user_input: str, data_content: Optional[str] = None, 
                           dict_content: Optional[str] = None) -> str:
        """Start conversation response"""
        context = self._get_relevant_context(user_input)
        if data_content:
            context += f"\n\nDATABASE REFERENCE:\n{data_content}"
        if dict_content:
            context += f"\n\nDICTIONARY REFERENCE:\n{dict_content}"
        
        prompt = f"AI Role: {ai_role}\nUser Role: {user_role}\nEvent: {event_name}\nDescription: {event_desc}\nUser Input: {user_input}"
        if context:
            prompt += f"\n\nCONTEXT:\n{context}"
        
        self.update_history("user", prompt, "BERT")
        response = self._generate_response(user_input, context)
        self.update_history("assistant", response, "BERT")
        
        return response
    
    def chat_continue_response(self, event_name: str, event_desc: str, user_role: str,
                              ai_role: str, user_input: str, data_content: Optional[str] = None,
                              dict_content: Optional[str] = None) -> str:
        """Continue conversation response"""
        return self.chat_start_response(event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content) 