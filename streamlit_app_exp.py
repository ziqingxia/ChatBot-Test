import streamlit as st
import os
import yaml
import json
import torch
from datetime import datetime
import tempfile
import shutil
import glob
import io
import re

from chatbot import ChatBot, InvalidAPIKeyError
from database_web import RAGKnowledgeBase
from fewshot import InContextLearner
from utils.streamlit_utils import process_uploaded_pdf, validate_pdf_file, get_database_info

# Import testing functionality
try:
    from scenario_testing import ScenarioTester, create_bert_chatbot, create_pure_bert_chatbot
    TESTING_AVAILABLE = True
except ImportError as e:
    TESTING_AVAILABLE = False
    TESTING_IMPORT_ERROR = e

# Check if PDF processing is available
from utils.streamlit_utils import PDF_PROCESSING_AVAILABLE

# === BertChat imports and setup ===
bert_available = True
bert_import_error = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    from sentence_transformers import SentenceTransformer
except Exception as e:
    bert_available = False
    bert_import_error = e
    st.warning(f"BertChat dependency import failed: {e}")

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.7rem;
        margin-bottom: 1.2rem;
        display: flex;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    .chat-message.user {
        background-color: #1565c0;
        color: #fff;
        border-left: 6px solid #003c8f;
        border-right: 2px solid #003c8f;
    }
    .chat-message.assistant {
        background-color: #ffe082;
        color: #222;
        border-left: 6px solid #ffb300;
        border-right: 2px solid #ffb300;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- API Key Input Section ---
st.sidebar.header("🔑 OpenAI API Key")
user_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    value=st.session_state.get("user_api_key", ""),
    help="Your key is only stored in this session and never sent anywhere else."
)

col1, col2 = st.sidebar.columns(2)
use_clicked = col1.button("Use this key")
stop_clicked = col2.button("Stop using this key")

if use_clicked:
    st.session_state["user_api_key"] = user_api_key
    os.environ["OPENAI_API_KEY"] = user_api_key
    st.session_state["api_key_cleared"] = False
    st.sidebar.success("API key set for this session!")

if stop_clicked:
    if "user_api_key" in st.session_state:
        del st.session_state["user_api_key"]
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    st.session_state["api_key_cleared"] = True
    st.sidebar.info("API key cleared for this session.")

# Status indicator
if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
    st.sidebar.markdown("🟢 **API Key is set for this session.**")
else:
    st.sidebar.markdown("🔴 **No API Key set.**")

# --- Prevent app from running if no API key is set ---
if ("user_api_key" not in st.session_state or not st.session_state["user_api_key"]):
    if st.session_state.get("api_key_cleared", False):
        st.info("You have cleared your API key. Please enter a new OpenAI API key in the sidebar to start a new conversation.")
        st.session_state["api_key_cleared"] = False
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to use the app.")
    st.stop()

# --- Unified Scenario/Event Selection ---
# Load scenarios/events from conversations.json
with open("samples/conversations.json", "r", encoding="utf-8") as f:
    conversations_data = json.load(f)

# Extract event names and descriptions
if isinstance(conversations_data, list):
    scenario_event_options = [item["event"] for item in conversations_data if "event" in item]
    scenario_event_descs = {item["event"]: item.get("description", "") for item in conversations_data if "event" in item}
else:
    scenario_event_options = []
    scenario_event_descs = {}

# --- Sidebar Mode Selection ---
st.sidebar.header("Testing/Evaluation")
if 'testing_selected' not in st.session_state:
    st.session_state['testing_selected'] = "None"
st.session_state['testing_selected'] = st.sidebar.radio("Select Evaluation Mode:", ["None", "Testing"], index=0)
testing_selected = st.session_state['testing_selected']

# --- Conditionally Render Sidebar Sections ---
if testing_selected != "Testing":
    st.sidebar.header("Scenario/Event Selection")
    st.sidebar.selectbox("Choose a scenario/event:", scenario_event_options, key="selected_scenario_event")
    levels = ["Basic", "Intermediate", "Advanced"]
    st.sidebar.selectbox("Proficiency Level:", levels, key="selected_level")
    st.sidebar.header("User Mode")
    st.sidebar.radio("Select your role:", ["Trainee", "Trainer"], key="user_mode")
    st.sidebar.header("Chatbot Selector")
    chatbot_choice = st.sidebar.radio("Choose Chatbot App:", ["OpenAI", "Hybrid", "BERT"], index=0)
else:
    chatbot_choice = None

# --- Scenario/Event Description ---
def get_scenario_event_description(event_name):
    return scenario_event_descs.get(event_name, f"Scenario/Event: {event_name} (no description available)")

# --- User Mode Toggle ---
# This block is now conditionally rendered based on testing_selected
# if testing_selected != "Testing":
#     st.sidebar.header("User Mode")
#     st.sidebar.radio("Select your role:", ["Trainee", "Trainer"], key="user_mode")
#     user_mode = st.session_state["user_mode"]

# Show current mode at the top of the main area
# This block is now conditionally rendered based on testing_selected
# if testing_selected != "Testing":
#     st.markdown(f"""
#     <div style='
#         background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
#         color: #fff;
#         padding: 1.2rem 1rem 1.2rem 1.2rem;
#         border-radius: 0.7rem;
#         margin-bottom: 1.2rem;
#         font-size: 1.2rem;
#         font-weight: 700;
#         box-shadow: 0 2px 8px rgba(24,90,157,0.15);
#     '>
#         Current Mode: {user_mode}
#     </div>
#     """, unsafe_allow_html=True)

@st.cache_resource
def load_config():
    """Load configuration file"""
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Do NOT set the API key from config, only use user input
    return config

@st.cache_resource
def load_rag_system(config):
    """Load RAG database and dictionary"""
    try:
        # Load RAG database
        rag_database = RAGKnowledgeBase(config, config['DATABASE']['ROOT_PATH'])
        
        # Load RAG dictionary
        rag_dictionary = RAGKnowledgeBase(config, config['DICTIONARY']['ROOT_PATH'])
        
        # Load phrases knowledge
        knowledge_phrases = RAGKnowledgeBase(
            config, 
            config['REFINE_KNOWLEDGE']['PHRASE_PATH'], 
            database_names=config['REFINE_KNOWLEDGE']['PHRASE_NAME']
        )
        
        # Load in-context learner
        context_learner = InContextLearner(config)
        
        return rag_database, rag_dictionary, knowledge_phrases, context_learner
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None, None, None, None

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    if 'current_event' not in st.session_state:
        st.session_state.current_event = None
    
    if 'current_event_desc' not in st.session_state:
        st.session_state.current_event_desc = None
    
    if 'ai_role' not in st.session_state:
        st.session_state.ai_role = None
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

def display_chat_message(role, content):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div style="flex-grow: 1;">
                <strong>You:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div style="flex-grow: 1;">
                <strong>Assistant:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

def build_system_prompt(scenario_desc, level):
    return f"""
You are an expert communications trainer for {level} level.
Scenario: {scenario_desc}

IMPORTANT: The AI Role and User Role provided are for context only. Do NOT require the user's message to match these names, callsigns, or roles. Do NOT penalize for any differences in names, callsigns, roles, numbers, or locations.

When scoring, ONLY evaluate:
- Correct use of protocol and required phrases (such as 'over')
- Message structure and clarity

If the message structure and required phrases are correct, score 100, regardless of the names, callsigns, or roles used.

If the score is 100, simply acknowledge correctness and continue the conversation. Do NOT provide corrections or protocol explanations.

If the score is less than 100, provide corrections, the score, and guidance on what to improve.

Examples:
- If the scenario uses 'TPO Tan' and 'Main Line Zero One', but the trainee says 'TPO Ho' and 'Main Line Zero Two', do NOT penalize for this difference. As long as the protocol and structure are correct, the score should be 100.
- If the user uses a different role or callsign, but the message is otherwise correct, do NOT penalize.

Evaluate the trainee's message for protocol adherence (callsigns, "over", clarity, etc.).
- If correct, reply "Correct" and give a score of 100.
- If incorrect, explain the mistake, provide the correct protocol, and give a score out of 100.
- Always provide a score and a brief reason.
"""

def start_new_conversation(event_name, event_desc, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config):
    """Start a new conversation"""
    st.session_state.current_event = event_name
    st.session_state.current_event_desc = event_desc
    st.session_state.ai_role = ai_role
    st.session_state.user_role = user_role
    st.session_state.chatbot = chatbot
    st.session_state.conversation_started = True
    st.session_state.messages = []
    
    # Set chat type with custom system prompt
    system_prompt = build_system_prompt(scenario_event_desc, selected_level)
    chatbot.set_chat_type(chat_type="conversation")
    chatbot.update_history("system", system_prompt, config['MODEL_TYPES']['QA_MODEL'])
    
    # Check if user starts or AI starts
    context_learner = InContextLearner(config)
    examples, is_user_start, ai_starter = context_learner.get_incontext_examples(event_name)
    
    if not is_user_start and ai_starter:
        # AI starts the conversation
        try:
            response = chatbot.chat_start_conversation(event_name, event_desc, user_role, ai_role, ai_starter)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        except InvalidAPIKeyError:
            st.session_state["api_key_cleared"] = True
            if "user_api_key" in st.session_state:
                del st.session_state["user_api_key"]
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            st.session_state.messages = [{
                "role": "assistant",
                "content": "🚫 Your API key is invalid. Please enter a valid OpenAI API key in the sidebar to continue."
            }]
            st.session_state.conversation_started = False
            st.rerun()

def strip_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def extract_placeholders_from_text(text):
    # Find all {PLACEHOLDER} in the text
    return set(re.findall(r'\{([A-Z0-9_]+)\}', text))

def get_current_scenario_turn():
    # Count user messages to determine which turn in the scenario we're on
    user_msgs = [m for m in st.session_state.messages if m['role'] == 'user']
    return len(user_msgs) - 1  # -1 because the current user input hasn't been appended yet

def update_placeholder_values(user_input):
    if 'placeholder_values' not in st.session_state:
        st.session_state['placeholder_values'] = {}
    scenario = next((s for s in st.session_state.get('scenarios', []) if s['event'] == st.session_state.current_event), None)
    if not scenario:
        return
    turn_idx = get_current_scenario_turn()
    # Find the next user utterance with placeholders
    user_turns = [i for i, t in enumerate(scenario['conversation']) if 'users' in t]
    if turn_idx < len(user_turns):
        scenario_turn = scenario['conversation'][user_turns[turn_idx]]
        template = scenario_turn['utterance']
        # Build a regex pattern from the template
        pattern = re.escape(template)
        for ph in extract_placeholders_from_text(template):
            pattern = pattern.replace(re.escape('{' + ph + '}'), r'(?P<' + ph + r'>.+?)')
        match = re.match(pattern, user_input)
        if match:
            for ph, value in match.groupdict().items():
                st.session_state['placeholder_values'][ph] = value.strip()

def substitute_placeholders(text):
    # Replace all {PLACEHOLDER} in text with values from session_state if available
    def repl(match):
        key = match.group(1)
        return st.session_state.get('placeholder_values', {}).get(key, match.group(0))
    return re.sub(r'\{([A-Z0-9_]+)\}', repl, text)

def clean_response(text):
    import re
    # Remove code blocks (triple backticks and content)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code (single backticks)
    text = re.sub(r"`[^`]*`", "", text)
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove HTML entities (e.g., &nbsp;, &lt;, &gt;)
    text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)
    # Remove lines that are just whitespace or empty after cleaning
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Remove lines that are just HTML tags (paranoia)
    lines = [line for line in lines if not re.match(r"^<[^>]+>$", line)]
    # Remove lines that are just HTML entities (paranoia)
    lines = [line for line in lines if not re.match(r"^&[a-zA-Z0-9#]+;$", line)]
    return "\n".join(lines).strip()

def process_user_message(user_input, chatbot, rag_database, rag_dictionary, knowledge_phrases, config):
    """Process user message and generate response"""
    event_name = st.session_state.current_event
    event_desc = st.session_state.current_event_desc
    ai_role = st.session_state.ai_role
    user_role = st.session_state.user_role
    
    # Search knowledge
    search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
    data_content = rag_database.search_knowledge(search_key, prefix="RAG Database", topk=config['SEARCH']['TOPK'])
    dict_content = rag_dictionary.search_knowledge(search_key, prefix="RAG Dictionary", topk=config['SEARCH']['TOPK'])
    
    # Generate response
    try:
        if not st.session_state.messages:
            # First message
            response = chatbot.chat_start_response(event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content)
        else:
            # Continue conversation
            response = chatbot.chat_continue_response(event_name, event_desc, user_role, ai_role, user_input, data_content, dict_content)
        # Try to extract score and feedback from response
        score = None
        feedback = response
        score_match = re.search(r"score\s*[:=]?\s*(\d{1,3})", response, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        # Display score and feedback, stripping HTML tags
        if score is not None:
            feedback = re.sub(r"score\s*[:=]?\s*\d{1,3}", "", response, flags=re.IGNORECASE).strip()
            feedback = strip_html_tags(feedback)
            feedback = substitute_placeholders(feedback)
            if score == 100:
                st.session_state.messages.append({"role": "assistant", "content": feedback})
            else:
                st.session_state.messages.append({"role": "assistant", "content": f"**Score:** {score}/100\n\n{feedback}"})
        else:
            feedback = strip_html_tags(feedback)
            feedback = substitute_placeholders(feedback)
            st.session_state.messages.append({"role": "assistant", "content": feedback})
        return ""
    except InvalidAPIKeyError:
        st.session_state["api_key_cleared"] = True
        if "user_api_key" in st.session_state:
            del st.session_state["user_api_key"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚫 Your API key is invalid. Please enter a valid OpenAI API key in the sidebar to continue."
        })
        st.session_state.conversation_started = False
        st.rerun()

def run_testing_app():
    """Testing page for scenario evaluation"""
    if not TESTING_AVAILABLE:
        st.error(f"Testing functionality not available. Import error: {TESTING_IMPORT_ERROR}")
        st.info("Please ensure scenario_testing.py and bert_chatbot.py are in the same directory.")
        return
    
    st.title("🧪 Chatbot Testing System")
    st.markdown("Compare OpenAI, Hybrid, and BERT chatbot performance across predefined railway communication scenarios.")
    
    # Load configuration
    config = load_config()
    
    # Initialize tester
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
    
    # Display selected scenarios
    st.sidebar.header("Selected Scenarios")
    for level in test_levels:
        if level in tester.scenarios:
            st.sidebar.markdown(f"**{level}:**")
            for scenario_id, scenario_data in tester.scenarios[level].items():
                st.sidebar.markdown(f"- {scenario_id}: {scenario_data['name']}")
    
    # Run tests button
    if st.button("🚀 Run All Tests", type="primary"):
        with st.spinner("Initializing chatbots..."):
            # Initialize OpenAI chatbot
            openai_chatbot = ChatBot(config)
            
            # Initialize Hybrid (was BertChat) if available
            hybrid_chatbot = None
            try:
                hybrid_chatbot = create_bert_chatbot(config)
                if hybrid_chatbot:
                    st.success("Hybrid initialized successfully")
                else:
                    st.warning("Hybrid initialization failed")
            except Exception as e:
                st.warning(f"Hybrid initialization failed: {e}")
            
            # Initialize BERT (was Pure BERT) chatbot if available
            bert_chatbot = None
            try:
                bert_chatbot = create_pure_bert_chatbot(config)
                if bert_chatbot:
                    st.success("BERT chatbot initialized successfully")
                else:
                    st.warning("BERT chatbot initialization failed")
            except Exception as e:
                st.warning(f"BERT chatbot initialization failed: {e}")
        
        # Run tests
        with st.spinner("Running tests..."):
            results = tester.run_all_scenarios(
                openai_chatbot, 
                hybrid_chatbot, 
                bert_chatbot,
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
        
        # Download button (CSV only)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=df.to_csv(index=False),
            file_name=results_file.replace('.json', '.csv'),
            mime="text/csv"
        )
    
    # Show scenario details
    st.markdown("## Test Scenarios")
    
    for level in test_levels:
        if level in tester.scenarios:
            st.markdown(f"### {level} Level")
            
            for scenario_id, scenario_data in tester.scenarios[level].items():
                with st.expander(f"{scenario_id} - {scenario_data['name']}"):
                    st.write("**Conversation Flow:**")
                    for i, turn in enumerate(scenario_data["conversation"]):
                        if turn["role"] == "user":
                            st.write(f"{i+1}. **User:** {turn['message']}")
                        elif turn["role"] == "expected":
                            st.write(f"{i+1}. **Expected Response:** {turn['message']}")

def should_display_code_block(code):
    import re
    code = code.strip()
    if not code:
        return False
    # Filter out code blocks that are just HTML tags (e.g., </div>)
    if re.fullmatch(r"</?[^>]+>", code):
        return False
    return True

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Load RAG system
    rag_database, rag_dictionary, knowledge_phrases, context_learner = load_rag_system(config)
    
    if rag_database is None:
        st.error("Failed to load RAG system. Please check your configuration and data files.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot System</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Event selection
        if not st.session_state.conversation_started:
            event_types, event_descs = context_learner.get_types()
            
            event_name = st.session_state['selected_scenario_event']
            event_desc = scenario_event_descs.get(event_name, "")
            ai_role, user_role = context_learner.get_roles(event_name)
            
            st.info(f"**Event:** {event_name}\n\n**Description:** {event_desc}\n\n**AI Role:** {ai_role}\n**User Role:** {user_role}")
            
            if st.button("Start Conversation", type="primary"):
                chatbot = ChatBot(config)
                start_new_conversation(event_name, event_desc, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
        
        # Conversation controls
        if st.session_state.conversation_started:
            st.subheader("Conversation Controls")
            
            if st.button("New Conversation"):
                st.session_state.conversation_started = False
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Clear History"):
                st.session_state.messages = []
                st.rerun()
            
            # Display current event info
            st.info(f"**Current Event:** {st.session_state.current_event}\n\n**AI Role:** {st.session_state.ai_role}\n**User Role:** {st.session_state.user_role}")
        
        # End Conversation and Export
        if st.sidebar.button("End Conversation & Export"):
            if st.session_state.get("messages"):
                # Save as JSON
                conversation_json = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.get('selected_scenario_event','scenario')}_{st.session_state.get('selected_level','level')}.json",
                    mime="application/json"
                )
                st.success("Conversation ready for download!")
                # Optionally, clear the conversation
                st.session_state["messages"] = []
                st.session_state["conversation_started"] = False
            else:
                st.sidebar.info("No conversation to export.")
    
    # Main chat area
    if st.session_state.conversation_started:
        # Check for quick test mode
        if st.session_state.get('show_quick_test', False):
            st.markdown("## 🧪 Quick Test Mode")
            st.info("Running a quick test of the current scenario...")
            
            # Initialize testing
            if TESTING_AVAILABLE:
                tester = ScenarioTester()
                config = load_config()
                
                # Find current scenario in test scenarios
                current_scenario_name = st.session_state.get('current_event', '')
                test_scenario = None
                
                for level, scenarios in tester.scenarios.items():
                    for scenario_id, scenario_data in scenarios.items():
                        if scenario_data['name'].lower() in current_scenario_name.lower():
                            test_scenario = scenario_data
                            break
                    if test_scenario:
                        break
                
                if test_scenario:
                    # Run quick test
                    with st.spinner("Running quick test..."):
                        openai_chatbot = ChatBot(config)
                        
                        try:
                            hybrid_chatbot = create_bert_chatbot(config)
                            if hybrid_chatbot:
                                st.success("Hybrid available")
                            else:
                                hybrid_chatbot = None
                                st.warning("Hybrid not available")
                        except:
                            hybrid_chatbot = None
                            st.warning("Hybrid not available")
                        
                        try:
                            bert_chatbot = create_pure_bert_chatbot(config)
                            if bert_chatbot:
                                st.success("BERT available")
                            else:
                                bert_chatbot = None
                                st.warning("BERT not available")
                        except:
                            bert_chatbot = None
                            st.warning("BERT not available")
                        
                        # Run single test
                        openai_results = tester.run_scenario(openai_chatbot, test_scenario, "quick_test")
                        
                        # Calculate scores
                        openai_scores = [turn["score"] for turn in openai_results if turn["score"] is not None]
                        openai_avg = sum(openai_scores) / len(openai_scores) if openai_scores else 0
                        
                        st.success(f"Quick Test Results:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("OpenAI Score", f"{openai_avg:.1f}/10")
                        
                        if hybrid_chatbot:
                            bertchat_results = tester.run_scenario(hybrid_chatbot, test_scenario, "quick_test")
                            bertchat_scores = [turn["score"] for turn in bertchat_results if turn["score"] is not None]
                            bertchat_avg = sum(bertchat_scores) / len(bertchat_scores) if bertchat_scores else 0
                            with col2:
                                st.metric("Hybrid Score", f"{bertchat_avg:.1f}/10")
                        
                        if bert_chatbot:
                            pure_bert_results = tester.run_scenario(bert_chatbot, test_scenario, "quick_test")
                            pure_bert_scores = [turn["score"] for turn in pure_bert_results if turn["score"] is not None]
                            pure_bert_avg = sum(pure_bert_scores) / len(pure_bert_scores) if pure_bert_scores else 0
                            with col3:
                                st.metric("BERT Score", f"{pure_bert_avg:.1f}/10")
                        
                        # Show detailed results
                        with st.expander("View Test Details"):
                            for turn in openai_results:
                                if turn["user_message"]:
                                    st.write(f"**User:** {turn['user_message']}")
                                if turn["chatbot_response"]:
                                    st.write(f"**Response:** {turn['chatbot_response']}")
                                if turn["expected_response"]:
                                    st.write(f"**Expected:** {turn['expected_response']}")
                                if turn["score"] is not None:
                                    st.write(f"**Score:** {turn['score']}/10")
                                st.write("---")
                
                if st.button("← Back to Chat"):
                    st.session_state['show_quick_test'] = False
                    st.rerun()
            else:
                st.error("Testing functionality not available")
                if st.button("← Back to Chat"):
                    st.session_state['show_quick_test'] = False
                    st.rerun()
        else:
            # Display chat messages
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
            
            # Chat input
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Process message and get response
                with st.spinner("Generating response..."):
                    try:
                        update_placeholder_values(user_input)
                        response = process_user_message(user_input, st.session_state.chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
                        st.session_state.messages.append({"role": "assistant", "content": clean_response(response)})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    else:
        # Welcome message
        st.info("👈 Please select an event type from the sidebar to start a conversation.")
        
        # Display available events
        if context_learner:
            event_types, event_descs = context_learner.get_types()
            
            st.subheader("Available Event Types")
            for i, (event_type, event_desc) in enumerate(zip(event_types, event_descs)):
                with st.expander(f"{i+1}. {event_type}"):
                    st.write(f"**Description:** {event_desc}")
                    ai_role, user_role = context_learner.get_roles(event_type)
                    st.write(f"**AI Role:** {ai_role}")
                    st.write(f"**User Role:** {user_role}")

def bertchat_main():
    # Check for BERT/LangChain dependencies
    if not bert_available:
        st.warning(f"Hybrid Chatbot requires 'langchain' and 'sentence-transformers'. Please install them to use this feature. ImportError: {bert_import_error}")
        return

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Load RAG system
    rag_database, rag_dictionary, knowledge_phrases, context_learner = load_rag_system(config)
    
    if rag_database is None:
        st.error("Failed to load RAG system. Please check your configuration and data files.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Hybrid Chatbot (BERT + LangChain Augmented)</h1>', unsafe_allow_html=True)

    # === BERT + LangChain Vector Store Setup ===
    @st.cache_resource
    def build_bert_faiss_store():
        # Gather all scenario/context chunks
        docs = []
        metadatas = []
        for item in rag_database.database.values():
            for meta, content in zip(item['meta'], item['content']):
                docs.append(content)
                metadatas.append({'meta': meta})
        # Use a stronger embedding model
        embedder = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        # Improved chunking
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        # Optionally, re-chunk docs here if needed (not shown in original code)
        # Build FAISS vector store
        vectorstore = FAISS.from_texts(docs, embedder, metadatas=metadatas)
        return vectorstore

    faiss_store = build_bert_faiss_store()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Event selection
        if not st.session_state.get('bertchat_conversation_started', False):
            event_types, event_descs = context_learner.get_types()
            
            event_name = st.session_state['selected_scenario_event']
            event_desc = scenario_event_descs.get(event_name, "")
            ai_role, user_role = context_learner.get_roles(event_name)
            
            st.info(f"**Event:** {event_name}\n\n**Description:** {event_desc}\n\n**AI Role:** {ai_role}\n**User Role:** {user_role}")
            
            if st.button("Start Conversation", type="primary", key="bertchat_start_convo"):
                chatbot = ChatBot(config)
                start_new_conversation(event_name, event_desc, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
                st.session_state['bertchat_conversation_started'] = True
        
        # Conversation controls
        if st.session_state.get('bertchat_conversation_started', False):
            st.subheader("Conversation Controls")
            
            if st.button("New Conversation", key="bertchat_new_convo"):
                st.session_state['bertchat_conversation_started'] = False
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Clear History", key="bertchat_clear_history"):
                st.session_state.messages = []
                st.rerun()
            
            # Display current event info
            st.info(f"**Current Event:** {st.session_state.current_event}\n\n**AI Role:** {st.session_state.ai_role}\n**User Role:** {st.session_state.user_role}")
        
        # End Conversation and Export
        if st.sidebar.button("End Conversation & Export", key="bertchat_export"):
            if st.session_state.get("messages"):
                # Save as JSON
                conversation_json = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=conversation_json,
                    file_name=f"hybrid_conversation_{st.session_state.get('selected_scenario_event','scenario')}_{st.session_state.get('selected_level','level')}.json",
                    mime="application/json"
                )
                st.success("Conversation ready for download!")
                # Optionally, clear the conversation
                st.session_state["messages"] = []
                st.session_state["conversation_started"] = False
            else:
                st.sidebar.info("No conversation to export.")
    
    # Main chat area
    if st.session_state.get('bertchat_conversation_started', False):
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...", key="bertchat_chat_input")
        
        if user_input:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # === BERT-powered retrieval ===
            # Embed user input and retrieve top-k relevant context chunks (increase k)
            docs_and_scores = faiss_store.similarity_search_with_score(user_input, k=8)
            retrieved_context = "\n".join([doc.page_content for doc, score in docs_and_scores])
            # Improved prompt engineering
            prompt = (
                "You are a helpful assistant. Use ONLY the information below to answer the user's question as accurately as possible.\n\n"
                f"Context:\n{retrieved_context}\n\n"
                f"User: {user_input}\n"
                "Assistant:"
            )
            with st.spinner("Generating response..."):
                try:
                    update_placeholder_values(user_input)
                    # Pass improved prompt to process_user_message (if possible)
                    response = process_user_message(prompt, st.session_state.chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
                    # Substitute placeholders with user-provided values
                    response = substitute_placeholders(response)
                    response = clean_response(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        # Welcome message
        st.info("👈 Please select an event type from the sidebar to start a conversation.")
        
        # Display available events
        if context_learner:
            event_types, event_descs = context_learner.get_types()
            
            st.subheader("Available Event Types")
            for i, (event_type, event_desc) in enumerate(zip(event_types, event_descs)):
                with st.expander(f"{i+1}. {event_type}"):
                    st.write(f"**Description:** {event_desc}")
                    ai_role, user_role = context_learner.get_roles(event_type)
                    st.write(f"**AI Role:** {ai_role}")
                    st.write(f"**User Role:** {user_role}")

def run_pure_bert_chatbot():
    """Run the Pure BERT chatbot (local, rule-based)"""
    try:
        from scenario_testing import create_pure_bert_chatbot
        from bert_chatbot import BertChatBot
    except ImportError as e:
        st.error(f"Pure BERT dependencies not found: {e}")
        st.info("Please ensure sentence-transformers and scikit-learn are installed.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Load RAG system for context
    rag_database, rag_dictionary, knowledge_phrases, context_learner = load_rag_system(config)
    
    if rag_database is None:
        st.error("Failed to load RAG system. Please check your configuration and data files.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Pure BERT Chatbot (Local)</h1>', unsafe_allow_html=True)
    st.info("Pure BERT chatbot uses only local BERT embeddings with rule-based responses. No external API calls required.")
    
    # Initialize Pure BERT chatbot
    try:
        pure_bert_chatbot = create_pure_bert_chatbot(config)
        if pure_bert_chatbot:
            st.success("✅ Pure BERT chatbot initialized successfully")
        else:
            st.error("❌ Failed to initialize Pure BERT chatbot")
            return
    except Exception as e:
        st.error(f"❌ Error initializing Pure BERT chatbot: {e}")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Event selection
        if not st.session_state.get('pure_bert_conversation_started', False):
            event_types, event_descs = context_learner.get_types()
            
            event_name = st.session_state['selected_scenario_event']
            event_desc = scenario_event_descs.get(event_name, "")
            ai_role, user_role = context_learner.get_roles(event_name)
            
            st.info(f"**Event:** {event_name}\n\n**Description:** {event_desc}\n\n**AI Role:** {ai_role}\n**User Role:** {user_role}")
            
            if st.button("Start Conversation", type="primary", key="pure_bert_start_convo"):
                start_new_conversation(event_name, event_desc, ai_role, user_role, pure_bert_chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
                st.session_state['pure_bert_conversation_started'] = True
        
        # Conversation controls
        if st.session_state.get('pure_bert_conversation_started', False):
            st.subheader("Conversation Controls")
            
            if st.button("New Conversation", key="pure_bert_new_convo"):
                st.session_state['pure_bert_conversation_started'] = False
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Clear History", key="pure_bert_clear_history"):
                st.session_state.messages = []
                st.rerun()
            
            # Display current event info
            st.info(f"**Current Event:** {st.session_state.current_event}\n\n**AI Role:** {st.session_state.ai_role}\n**User Role:** {st.session_state.user_role}")
        
        # End Conversation and Export
        if st.sidebar.button("End Conversation & Export", key="pure_bert_export"):
            if st.session_state.get("messages"):
                # Save as JSON
                conversation_json = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=conversation_json,
                    file_name=f"pure_bert_conversation_{st.session_state.get('selected_scenario_event','scenario')}_{st.session_state.get('selected_level','level')}.json",
                    mime="application/json"
                )
                st.success("Conversation ready for download!")
                # Optionally, clear the conversation
                st.session_state["messages"] = []
                st.session_state["conversation_started"] = False
            else:
                st.sidebar.info("No conversation to export.")
    
    # Main chat area
    if st.session_state.get('pure_bert_conversation_started', False):
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...", key="pure_bert_chat_input")
        
        if user_input:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Generating response..."):
                try:
                    # Use the Pure BERT chatbot's test method
                    response = pure_bert_chatbot.test(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        # Welcome message
        st.info("👈 Please select an event type from the sidebar to start a conversation.")
        
        # Display available events
        if context_learner:
            event_types, event_descs = context_learner.get_types()
            
            st.subheader("Available Event Types")
            for i, (event_type, event_desc) in enumerate(zip(event_types, event_descs)):
                with st.expander(f"{i+1}. {event_type}"):
                    st.write(f"**Description:** {event_desc}")
                    ai_role, user_role = context_learner.get_roles(event_type)
                    st.write(f"**AI Role:** {ai_role}")
                    st.write(f"**User Role:** {user_role}")

# Sidebar app selector
# This block is now conditionally rendered based on testing_selected
# if testing_selected != "Testing":
#     st.sidebar.header("Chatbot Selector")
#     chatbot_choice = st.sidebar.radio("Choose Chatbot App:", ["OpenAI", "Hybrid", "BERT"], index=0)
# else:
#     chatbot_choice = None

# New section for Testing/Evaluation
# This block is now conditionally rendered based on testing_selected
# if testing_selected != "Testing":
#     st.sidebar.header("Testing/Evaluation")
#     st.session_state['testing_selected'] = st.sidebar.radio("Select Evaluation Mode:", ["None", "Testing"], index=0)
# else:
#     st.session_state['testing_selected'] = "Testing"

# testing_selected = st.session_state['testing_selected']

# Main UI logic
if testing_selected == "Testing":
    run_testing_app()
else:
    # Conditionally render scenario/event selection, user mode, and chatbot selector
    if testing_selected != "Testing":
        selected_scenario_event = st.session_state.get("selected_scenario_event", scenario_event_options[0] if scenario_event_options else "")
        selected_level = st.session_state.get("selected_level", "Basic")
        user_mode = st.session_state.get("user_mode", "Trainee")
        # Show scenario/event and level at the top of the chat area
        scenario_event_desc = scenario_event_descs.get(selected_scenario_event, f"Scenario/Event: {selected_scenario_event} (no description available)")
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, #1565c0 60%, #42a5f5 100%);
            color: #fff;
            padding: 1.2rem 1rem 1.2rem 1.2rem;
            border-radius: 0.7rem;
            margin-bottom: 1.2rem;
            font-size: 1.15rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(21,101,192,0.15);
        '>
            <div><span style="font-size:1.1rem;font-weight:700;">Scenario/Event:</span> {selected_scenario_event}</div>
            <div><span style="font-size:1.1rem;font-weight:700;">Level:</span> {selected_level}</div>
            <div><span style="font-size:1.1rem;font-weight:700;">Description:</span> {scenario_event_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
            color: #fff;
            padding: 1.2rem 1rem 1.2rem 1.2rem;
            border-radius: 0.7rem;
            margin-bottom: 1.2rem;
            font-size: 1.2rem;
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(24,90,157,0.15);
        '>
            Current Mode: {user_mode}
        </div>
        """, unsafe_allow_html=True)
    else:
        # If testing is selected, only show the Testing/Evaluation section
        st.sidebar.header("Testing/Evaluation")
        st.session_state['testing_selected'] = st.sidebar.radio("Select Evaluation Mode:", ["None", "Testing"], index=0)
        testing_selected = st.session_state['testing_selected']

    if chatbot_choice == "OpenAI":
        main()
    elif chatbot_choice == "Hybrid":
        bertchat_main()
    elif chatbot_choice == "BERT":
        run_pure_bert_chatbot() 