import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Tuple, Any, Optional
import yaml
import torch

# Import chatbot classes
from chatbot import ChatBot, InvalidAPIKeyError

# Try to import BERT components
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

class ScenarioTester:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the scenario tester with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scenarios = self._load_test_scenarios()
        self.results = {}
        
    def _load_test_scenarios(self) -> Dict[str, List[Dict]]:
        """Load the predefined test scenarios"""
        return {
            "Beginner": {
                "2.1": {
                    "name": "Name Recognition and Consistency",
                    "conversation": [
                        {"role": "user", "message": "Kendrik, this is TPO Lim message, over."},
                        {"role": "expected", "message": "Kendrik, send, over."},
                        {"role": "user", "message": "TPO Lim, Radio Check, over."},
                        {"role": "expected", "message": "Kendrik, Radio Check OK, over."},
                        {"role": "user", "message": "TPO Lim, Radio Check OK. out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "2.2": {
                    "name": "Different Phrasing Test",
                    "conversation": [
                        {"role": "user", "message": "Kendrik, this is TPO XX message, over."},
                        {"role": "expected", "message": "Kendrik, send, over."},
                        {"role": "user", "message": "TPO XX, How are you doing. Check your radio, over."},
                        {"role": "expected", "message": "Kendrik, Radio Check OK, over."},
                        {"role": "user", "message": "TPO XX, Have a nice day. Radio Check OK. out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "2.3": {
                    "name": "Control Set",
                    "conversation": [
                        {"role": "user", "message": "Kendrik, this is TPO XX message, over."},
                        {"role": "expected", "message": "Kendrik, send, over."},
                        {"role": "user", "message": "TPO XX, Radio Check, over."},
                        {"role": "expected", "message": "Kendrik, Radio Check OK, over."},
                        {"role": "user", "message": "TPO XX, Radio Check OK. out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                }
            },
            "Intermediate": {
                "6.1": {
                    "name": "Naming Recognition and Consistency",
                    "conversation": [
                        {"role": "expected", "message": "Main Line Zero Two, Send, over"},
                        {"role": "user", "message": "TPO Ho, I have completed my work and want to surrender my TOA, over"},
                        {"role": "expected", "message": "Main Line Zero Two, please proceed with your Line Clear message, over"},
                        {"role": "user", "message": "TPO Ho, All staff, material and equipment are off the track and tunnel and it is safe for train to run. I surrender my TOA 2020-00257, over"},
                        {"role": "expected", "message": "Main Line Zero Two, Line Clear Message Acknowledged, TOA 2020-00257 surrendered at 0415 hrs, over"},
                        {"role": "user", "message": "TPO Ho, TOA surrendered at 0415 hrs, over"},
                        {"role": "expected", "message": "Main Line Zero Two, that is correct, out"},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "6.2": {
                    "name": "Different Phrasing Test",
                    "conversation": [
                        {"role": "expected", "message": "Main Line Zero Two, Send, over"},
                        {"role": "user", "message": "TPO Ho, work done and want to move on, over"},
                        {"role": "expected", "message": "Main Line Zero Two, please proceed with your Line Clear message, over"},
                        {"role": "user", "message": "TPO Ho, everything done. I surrender my TOA 2020-00257, over"},
                        {"role": "expected", "message": "Main Line Zero Two, Line Clear Message Acknowledged, TOA 2020-00257 surrendered at 0415 hrs, over"},
                        {"role": "user", "message": "TPO Ho, TOA surrendered at 0415 hrs, over"},
                        {"role": "expected", "message": "Main Line Zero Two, that is correct, out"},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "6.3": {
                    "name": "Control Set",
                    "conversation": [
                        {"role": "expected", "message": "Main Line Zero Two, Send, over"},
                        {"role": "user", "message": "TPO Ho, I have completed my work and want to surrender my TOA, over"},
                        {"role": "expected", "message": "Main Line Zero Two, please proceed with your Line Clear message, over"},
                        {"role": "user", "message": "TPO Ho, All staff, material and equipment are off the track and tunnel and it is safe for train to run. I surrender my TOA 2020-00257, over"},
                        {"role": "expected", "message": "Main Line Zero Two, Line Clear Message Acknowledged, TOA 2020-00257 surrendered at 0415 hrs, over"},
                        {"role": "user", "message": "TPO Ho, TOA surrendered at 0415 hrs, over"},
                        {"role": "expected", "message": "Main Line Zero Two, that is correct, out"},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                }
            },
            "Advanced": {
                "14.1": {
                    "name": "Protocol Adherence",
                    "conversation": [
                        {"role": "user", "message": "Help me. Rail crack spotted near point Eight Three Three."},
                        {"role": "expected", "message": "Depot Zero, read back message, stop your train immediately and standby for further instructions, over."},
                        {"role": "user", "message": "I read back message, train stopped, standing by for further instructions."},
                        {"role": "expected", "message": "Depot Zero, roger, wait out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "14.2": {
                    "name": "Different Phrasing Test",
                    "conversation": [
                        {"role": "user", "message": "Emergency, Emergency. Rail crack spotted near point Eight Three Three, over."},
                        {"role": "expected", "message": "Depot Zero, read back message, stop your train immediately and standby for further instructions, over."},
                        {"role": "user", "message": "Car Zero Nine, okay whats next? over."},
                        {"role": "expected", "message": "Depot Zero, roger, wait out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                },
                "14.3": {
                    "name": "Protocol Adherence (Detailed)",
                    "conversation": [
                        {"role": "user", "message": "Emergency, Emergency, Depot Zero, this is Car Zero Nine at track Sierra One East. Rail crack spotted near point Eight Three Three, over."},
                        {"role": "expected", "message": "Depot Zero, read back message, stop your train immediately and standby for further instructions, over."},
                        {"role": "user", "message": "Car Zero Nine, I read back message, train stopped, standing by for further instructions, over."},
                        {"role": "expected", "message": "Depot Zero, roger, wait out."},
                        {"role": "expected", "message": "You have done responded to the SOP correctly. You have reached the end of the conversation"}
                    ]
                }
            }
        }
    
    def evaluate_response(self, expected: str, actual: str) -> int:
        """
        Evaluate chatbot response against expected response using BERT cosine similarity.
        Returns score from 1-10
        """
        # Lazy-load and cache the BERT model
        if not hasattr(self, '_bert_model'):
            from sentence_transformers import SentenceTransformer
            self._bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        model = self._bert_model

        expected_emb = model.encode([expected])[0]
        actual_emb = model.encode([actual])[0]

        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([expected_emb], [actual_emb])[0][0]
        # Scale similarity (range -1 to 1) to score 1-10
        # Clamp sim to [0, 1] for safety
        sim = max(0, min(1, sim))
        score = int(round(sim * 9 + 1))  # sim=1 -> 10, sim=0 -> 1
        return score
    
    def run_scenario(self, chatbot, scenario_data: Dict, scenario_name: str) -> List[Dict]:
        """Run a single scenario and return results"""
        results = []
        conversation = scenario_data["conversation"]
        
        # Clear chatbot history for fresh start
        chatbot.clear_history()
        
        for i, turn in enumerate(conversation):
            if turn["role"] == "user":
                # User message - chatbot should respond
                user_message = turn["message"]
                
                # Get chatbot response
                try:
                    response = chatbot.test(user_message, None, None)
                    results.append({
                        "turn": i + 1,
                        "user_message": user_message,
                        "chatbot_response": response,
                        "expected_response": None,
                        "score": None,
                        "notes": "User message - no scoring needed"
                    })
                except Exception as e:
                    results.append({
                        "turn": i + 1,
                        "user_message": user_message,
                        "chatbot_response": f"ERROR: {str(e)}",
                        "expected_response": None,
                        "score": None,
                        "notes": f"Error occurred: {str(e)}"
                    })
                    
            elif turn["role"] == "expected":
                # Expected chatbot response - evaluate previous response
                expected_response = turn["message"]
                
                if results and results[-1]["chatbot_response"] and not results[-1]["chatbot_response"].startswith("ERROR"):
                    actual_response = results[-1]["chatbot_response"]
                    score = self.evaluate_response(expected_response, actual_response)
                    
                    # Update the previous result with evaluation
                    results[-1]["expected_response"] = expected_response
                    results[-1]["score"] = score
                    results[-1]["notes"] = f"Scored: {score}/10"
                else:
                    results.append({
                        "turn": i + 1,
                        "user_message": None,
                        "chatbot_response": None,
                        "expected_response": expected_response,
                        "score": 0,
                        "notes": "No response to evaluate"
                    })
        
        return results
    
    def run_all_scenarios(self, openai_chatbot, bert_chatbot=None, pure_bert_chatbot=None, num_runs: int = 5) -> Dict:
        """Run all scenarios multiple times and return comprehensive results"""
        all_results = {}
        
        for level, scenarios in self.scenarios.items():
            all_results[level] = {}
            
            for scenario_id, scenario_data in scenarios.items():
                all_results[level][scenario_id] = {
                    "name": scenario_data["name"],
                    "openai_runs": [],
                    "bertchat_runs": [],
                    "pure_bert_runs": []
                }
                
                # Run OpenAI scenarios
                for run in range(num_runs):
                    st.write(f"Running {level} - {scenario_id} (OpenAI) - Run {run + 1}/{num_runs}")
                    try:
                        run_results = self.run_scenario(openai_chatbot, scenario_data, f"{level}_{scenario_id}")
                        all_results[level][scenario_id]["openai_runs"].append(run_results)
                    except Exception as e:
                        st.error(f"Error in OpenAI run {run + 1}: {str(e)}")
                        all_results[level][scenario_id]["openai_runs"].append([])
                
                # Run BertChat scenarios if available
                if bert_chatbot:
                    for run in range(num_runs):
                        st.write(f"Running {level} - {scenario_id} (BertChat) - Run {run + 1}/{num_runs}")
                        try:
                            run_results = self.run_scenario(bert_chatbot, scenario_data, f"{level}_{scenario_id}")
                            all_results[level][scenario_id]["bertchat_runs"].append(run_results)
                        except Exception as e:
                            st.error(f"Error in BertChat run {run + 1}: {str(e)}")
                            all_results[level][scenario_id]["bertchat_runs"].append([])
                
                # Run Pure BERT scenarios if available
                if pure_bert_chatbot:
                    for run in range(num_runs):
                        st.write(f"Running {level} - {scenario_id} (Pure BERT) - Run {run + 1}/{num_runs}")
                        try:
                            run_results = self.run_scenario(pure_bert_chatbot, scenario_data, f"{level}_{scenario_id}")
                            all_results[level][scenario_id]["pure_bert_runs"].append(run_results)
                        except Exception as e:
                            st.error(f"Error in Pure BERT run {run + 1}: {str(e)}")
                            all_results[level][scenario_id]["pure_bert_runs"].append([])
        
        return all_results
    
    def calculate_scores(self, results: Dict) -> pd.DataFrame:
        """Calculate and format scores for display"""
        score_data = []
        
        for level, scenarios in results.items():
            for scenario_id, scenario_data in scenarios.items():
                scenario_name = scenario_data["name"]
                
                # Calculate OpenAI scores
                openai_scores = []
                for run in scenario_data["openai_runs"]:
                    run_scores = [turn["score"] for turn in run if turn["score"] is not None]
                    if run_scores:
                        openai_scores.extend(run_scores)
                
                openai_avg = sum(openai_scores) / len(openai_scores) if openai_scores else 0
                
                # Calculate BertChat scores
                bertchat_scores = []
                for run in scenario_data["bertchat_runs"]:
                    run_scores = [turn["score"] for turn in run if turn["score"] is not None]
                    if run_scores:
                        bertchat_scores.extend(run_scores)
                
                bertchat_avg = sum(bertchat_scores) / len(bertchat_scores) if bertchat_scores else 0
                
                # Calculate Pure BERT scores
                pure_bert_scores = []
                for run in scenario_data["pure_bert_runs"]:
                    run_scores = [turn["score"] for turn in run if turn["score"] is not None]
                    if run_scores:
                        pure_bert_scores.extend(run_scores)
                
                pure_bert_avg = sum(pure_bert_scores) / len(pure_bert_scores) if pure_bert_scores else 0
                
                # Create row for this scenario
                row = {
                    "Level": level,
                    "Scenario": f"{scenario_id} - {scenario_name}",
                    "OpenAI Scores": openai_scores,
                    "OpenAI Total": round(openai_avg, 1),
                    "BertChat Scores": bertchat_scores,
                    "BertChat Total": round(bertchat_avg, 1),
                    "Pure BERT Scores": pure_bert_scores,
                    "Pure BERT Total": round(pure_bert_avg, 1)
                }
                score_data.append(row)
        
        return pd.DataFrame(score_data)
    
    def display_results_table(self, df: pd.DataFrame):
        """Display results in the requested table format"""
        st.markdown("## Test Results Summary")
        
        # Create a formatted table
        for level in df['Level'].unique():
            level_data = df[df['Level'] == level]
            st.markdown(f"### {level} Level")
            
            # Create table data
            table_data = []
            for _, row in level_data.iterrows():
                scenario_name = row['Scenario']
                openai_scores = row['OpenAI Scores']
                bertchat_scores = row['BertChat Scores']
                pure_bert_scores = row['Pure BERT Scores']
                
                # Format scores as strings
                openai_score_str = " | ".join([str(s) for s in openai_scores])
                bertchat_score_str = " | ".join([str(s) for s in bertchat_scores])
                pure_bert_score_str = " | ".join([str(s) for s in pure_bert_scores])
                
                table_data.append({
                    "Scenario": scenario_name,
                    "OpenAI": f"{openai_score_str}\nTotal: {row['OpenAI Total']}",
                    "BertChat": f"{bertchat_score_str}\nTotal: {row['BertChat Total']}",
                    "Pure BERT": f"{pure_bert_score_str}\nTotal: {row['Pure BERT Total']}"
                })
            
            # Display table
            if table_data:
                st.table(pd.DataFrame(table_data))
        
        # Overall summary
        st.markdown("### Overall Summary")
        overall_openai = df['OpenAI Total'].mean()
        overall_bertchat = df['BertChat Total'].mean()
        overall_pure_bert = df['Pure BERT Total'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("OpenAI Average Score", f"{overall_openai:.1f}/10")
        with col2:
            st.metric("BertChat Average Score", f"{overall_bertchat:.1f}/10")
        with col3:
            st.metric("Pure BERT Average Score", f"{overall_pure_bert:.1f}/10")

def create_bert_chatbot(config):
    """Create a BertChat chatbot using the same ChatBot class as the main app."""
    try:
        return ChatBot(config)
    except ImportError as e:
        st.error(f"BertChat dependencies not found: {e}")
        st.info("Please ensure langchain, sentence-transformers, and faiss-cpu are installed.")
        return None

def create_pure_bert_chatbot(config):
    """Create a pure BERT-based chatbot (local, rule-based)"""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        class PureBertChatBot:
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
                self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_pure_bert_main.json")
                self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_pure_bert_refine.json")
            
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
                                    with open(json_file, 'r', encoding="utf-8") as f:
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
                                    with open(json_file, 'r', encoding="utf-8") as f:
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
                self.history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_pure_bert_main.json")
                self.refine_history_path = os.path.join(self.config['CHATBOT']['HISTORY_PATH'], f"{self.chat_name}_pure_bert_refine.json")
            
            def update_history(self, role: str, prompt: str, model_type: str = "PureBERT"):
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
                # Rule-based approach with context matching
                if context:
                    return self._generate_contextual_response(user_input, context)
                else:
                    return self._generate_basic_response(user_input)
            
            def _generate_contextual_response(self, user_input: str, context: str) -> str:
                """Generate response using context"""
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
                
                self.update_history("user", prompt, "PureBERT")
                
                # Generate response
                response = self._generate_response(user_input, context)
                
                # Update history with response
                self.update_history("assistant", response, "PureBERT")
                
                print(f"====="*10)
                print(f"Pure BERT Response: {response}")
                return response
        
        return PureBertChatBot(config)
        
    except ImportError as e:
        st.error(f"Pure BERT dependencies not found: {e}")
        st.info("Please ensure sentence-transformers and scikit-learn are installed.")
        return None

def main():
    st.set_page_config(
        page_title="Chatbot Scenario Testing",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Chatbot Scenario Testing System")
    st.markdown("This system tests chatbot performance across predefined scenarios and compares OpenAI vs BERT implementations.")
    
    # Check for API key
    if "user_api_key" not in st.session_state or not st.session_state["user_api_key"]:
        st.error("Please set your OpenAI API key in the main app first.")
        st.stop()
    
    # Initialize testers
    tester = ScenarioTester()
    
    # Configuration
    st.sidebar.header("Test Configuration")
    num_runs = st.sidebar.slider("Number of test runs", 1, 10, 5)
    
    # Test levels selection
    st.sidebar.header("Test Levels")
    test_levels = st.sidebar.multiselect(
        "Select levels to test",
        ["Beginner", "Intermediate", "Advanced"],
        default=["Beginner", "Intermediate", "Advanced"]
    )
    
    # Filter scenarios based on selected levels
    filtered_scenarios = {level: scenarios for level, scenarios in tester.scenarios.items() 
                         if level in test_levels}
    
    # Display selected scenarios
    st.sidebar.header("Selected Scenarios")
    for level, scenarios in filtered_scenarios.items():
        st.sidebar.markdown(f"**{level}:**")
        for scenario_id, scenario_data in scenarios.items():
            st.sidebar.markdown(f"- {scenario_id}: {scenario_data['name']}")
    
    # Run tests button
    if st.button("🚀 Run All Tests", type="primary"):
        with st.spinner("Initializing chatbots..."):
            # Initialize OpenAI chatbot
            openai_chatbot = ChatBot(tester.config)
            
            # Initialize BERT chatbot if available
            bert_chatbot = None
            if BERT_AVAILABLE:
                try:
                    bert_chatbot = create_bert_chatbot(tester.config)
                    st.success("BERT chatbot initialized successfully")
                except Exception as e:
                    st.warning(f"BERT chatbot initialization failed: {e}")
            else:
                st.warning("BERT dependencies not available")
        
        # Initialize Pure BERT chatbot
        pure_bert_chatbot = None
        try:
            pure_bert_chatbot = create_pure_bert_chatbot(tester.config) # Pass tester.config here
            if pure_bert_chatbot:
                st.success("Pure BERT chatbot initialized successfully")
            else:
                st.warning("Pure BERT chatbot initialization failed")
        except Exception as e:
            st.warning(f"Pure BERT chatbot initialization failed: {e}")
        
        # Run tests
        with st.spinner("Running tests..."):
            results = tester.run_all_scenarios(
                openai_chatbot, 
                bert_chatbot, 
                pure_bert_chatbot,
                num_runs
            )
        
        # Calculate and display results
        df = tester.calculate_scores(results)
        tester.display_results_table(df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        st.success(f"Results saved to {results_file}")
        
        # Download button
        st.download_button(
            label="📥 Download Results",
            data=json.dumps(results, indent=2, default=str),
            file_name=results_file,
            mime="application/json"
        )

if __name__ == "__main__":
    main() 