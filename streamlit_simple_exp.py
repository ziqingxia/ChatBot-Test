import streamlit as st
import os
import yaml
import json
import torch
from datetime import datetime
import re
import importlib

# === CUDA Safety Setup ===
# Check for CUDA availability and handle torch imports safely
try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        st.info("CUDA is available. Using GPU acceleration.")
    else:
        st.info("CUDA is not available. Using CPU.")
except Exception as e:
    st.warning(f"Error checking CUDA availability: {e}")

# Import dependencies with error handling
try:
    from chatbot_exp import ChatBot, InvalidAPIKeyError
    from database_web import RAGKnowledgeBase  # Use web version for CUDA safety
    from fewshot_exp import InContextLearner
    DEPENDENCIES_AVAILABLE = True
except Exception as e:
    st.error(f"Failed to import required dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Safety Training ChatBot - Simple",
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
    .event-info {
        background: linear-gradient(90deg, #1565c0 60%, #42a5f5 100%);
        color: #fff;
        padding: 1.2rem;
        border-radius: 0.7rem;
        margin-bottom: 1.2rem;
        font-size: 1.15rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(21,101,192,0.15);
    }
    .persona-info {
        background: linear-gradient(90deg, #4caf50 60%, #81c784 100%);
        color: #fff;
        padding: 1rem;
        border-radius: 0.7rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(76,175,80,0.15);
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

@st.cache_resource
def load_config():
    """Load configuration file"""
    try:
        with open('configs/config_exp.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None

@st.cache_resource
def load_rag_system(config):
    """Load RAG database and dictionary with CUDA safety"""
    try:
        # Check CUDA availability before loading
        if torch.cuda.is_available():
            st.info("CUDA is available. Using GPU acceleration.")
        else:
            st.info("CUDA is not available. Using CPU.")
        
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
        st.error("This error might be related to CUDA/NVIDIA driver issues. Please check your GPU setup.")
        return None, None, None, None

def load_persona_prompts(persona_number):
    """Dynamically load persona prompts"""
    try:
        persona_module = importlib.import_module(f"prompts_persona{persona_number}")
        return persona_module
    except ImportError:
        st.error(f"Could not load persona {persona_number}")
        return None

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
    
    if 'current_event_obj' not in st.session_state:
        st.session_state.current_event_obj = None
    
    if 'current_event_conv' not in st.session_state:
        st.session_state.current_event_conv = None
    
    if 'current_event_point' not in st.session_state:
        st.session_state.current_event_point = None
    
    if 'current_event_que' not in st.session_state:
        st.session_state.current_event_que = None
    
    if 'ai_role' not in st.session_state:
        st.session_state.ai_role = None
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    
    if 'selected_persona' not in st.session_state:
        st.session_state.selected_persona = 1
    
    if 'intro_completed' not in st.session_state:
        st.session_state.intro_completed = False
    
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False

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

def clean_response(text):
    """Clean response text while preserving some formatting"""
    import re
    
    # Remove code blocks (triple backticks and content)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    
    # Convert markdown bold to HTML bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    
    # Convert markdown italic to HTML italic
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    
    # Convert markdown headers to HTML headers
    text = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    
    # Convert markdown lists to HTML lists
    text = re.sub(r"^\- (.*?)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^\* (.*?)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    
    # Add line breaks for better formatting
    text = re.sub(r"\n\n", r"<br><br>", text)
    text = re.sub(r"\n", r"<br>", text)
    
    # Remove inline code (single backticks) but keep the content
    text = re.sub(r"`([^`]*)`", r"<code>\1</code>", text)
    
    # Remove unwanted HTML tags but keep our formatting
    text = re.sub(r"<(?!strong|em|h1|h2|h3|li|br|code)[^>]+>", "", text)
    
    # Remove HTML entities but keep basic ones
    text = re.sub(r"&(?!nbsp|lt|gt|amp)[a-zA-Z0-9#]+;", "", text)
    
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text

def start_new_conversation(event_name, event_desc, event_obj, event_conv, event_point, event_que, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config, persona_module):
    """Start a new conversation based on main_exp.py logic"""
    st.session_state.current_event = event_name
    st.session_state.current_event_desc = event_desc
    st.session_state.current_event_obj = event_obj
    st.session_state.current_event_conv = event_conv
    st.session_state.current_event_point = event_point
    st.session_state.current_event_que = event_que
    st.session_state.ai_role = ai_role
    st.session_state.user_role = user_role
    st.session_state.chatbot = chatbot
    st.session_state.conversation_started = True
    st.session_state.messages = []
    st.session_state.intro_completed = False
    st.session_state.training_started = False
    
    # Set chat type
    chatbot.set_chat_type(chat_type="conversation")
    
    # Load persona prompts
    chatbot.set_persona_module(persona_module)
    
    # Start with intro (based on main_exp.py logic)
    try:
        response = chatbot.chat_intro(event_name, event_desc, event_obj, event_conv, user_role, ai_role, "")
        st.session_state.messages.append({"role": "assistant", "content": clean_response(response)})
        st.session_state.intro_completed = True
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

def process_user_message(user_input, chatbot, rag_database, rag_dictionary, knowledge_phrases, config):
    """Process user message based on main_exp.py logic"""
    event_name = st.session_state.current_event
    event_desc = st.session_state.current_event_desc
    event_obj = st.session_state.current_event_obj
    event_conv = st.session_state.current_event_conv
    event_point = st.session_state.current_event_point
    event_que = st.session_state.current_event_que
    ai_role = st.session_state.ai_role
    user_role = st.session_state.user_role
    
    # Check if user wants to break
    if user_input.lower() == 'break':
        st.session_state.conversation_started = False
        st.session_state.messages = []
        st.session_state.intro_completed = False
        st.session_state.training_started = False
        st.rerun()
        return "Conversation ended by user."
    
    # Handle intro phase (based on main_exp.py logic)
    if st.session_state.intro_completed and not st.session_state.training_started:
        if user_input.lower() == 'start':
            st.session_state.training_started = True
            # Start phase 1 (based on main_exp.py logic)
            response = chatbot.chat_start_phase1(event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role)
            return clean_response(response)
        else:
            return "Please type 'start' to begin the training, or 'break' to end the conversation."
    
    # Continue conversation (based on main_exp.py logic)
    if st.session_state.training_started:
        # Search knowledge (based on main_exp.py logic)
        search_key = f"Event: {event_name}\nDescription: {event_desc}\nai role: {ai_role}\nusers: {user_role}\nutterance: {user_input}"
        data_content = rag_database.search_knowledge(search_key, prefix="RAG Database", topk=config['SEARCH']['TOPK'])
        dict_content = rag_dictionary.search_knowledge(search_key, prefix="RAG Dictionary", topk=config['SEARCH']['TOPK'])
        
        # Continue conversation (based on main_exp.py logic)
        response = chatbot.chat_continue_phase1(event_name, event_desc, event_obj, event_point, event_conv, event_que, user_role, ai_role, user_input, data_content, dict_content)
        return clean_response(response)
    else:
        return "Please type 'start' to begin the training, or 'break' to end the conversation."

def main():
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        st.error("Required dependencies are not available. Please check your installation.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    if config is None:
        st.error("Failed to load configuration.")
        return
    
    # Load RAG system
    rag_database, rag_dictionary, knowledge_phrases, context_learner = load_rag_system(config)
    
    if rag_database is None:
        st.error("Failed to load RAG system. Please check your configuration and data files.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Safety Training ChatBot - Simple</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Persona selection
        st.subheader("🎭 Persona Selection")
        persona_options = {
            1: "Strict Instructor (Authoritative)",
            2: "Friendly Peer (Supportive)", 
            3: "AI Assistant (Structured)"
        }
        
        selected_persona = st.selectbox(
            "Choose your training persona:",
            options=list(persona_options.keys()),
            format_func=lambda x: persona_options[x],
            index=st.session_state.selected_persona - 1
        )
        
        if selected_persona != st.session_state.selected_persona:
            st.session_state.selected_persona = selected_persona
            if st.session_state.conversation_started:
                st.session_state.conversation_started = False
                st.session_state.messages = []
                st.rerun()
        
        # Display persona info
        persona_descriptions = {
            1: "Strict and authoritative trainer who emphasizes precision and discipline",
            2: "Friendly peer who provides encouragement and relatable tips",
            3: "Calm AI assistant focused on structured, step-by-step guidance"
        }
        
        st.markdown(f"""
        <div class="persona-info">
            <div><strong>Current Persona:</strong> {persona_options[selected_persona]}</div>
            <div><strong>Style:</strong> {persona_descriptions[selected_persona]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Event selection
        if not st.session_state.conversation_started:
            event_types, event_descs, event_objs, event_convs, event_points, event_ques = context_learner.get_types()
            
            # Event selection dropdown
            selected_event = st.selectbox(
                "Choose a training scenario:",
                event_types,
                format_func=lambda x: f"{x}",
                help="Select the type of safety event to practice"
            )
            
            if selected_event:
                event_idx = event_types.index(selected_event)
                event_desc = event_descs[event_idx]
                event_obj = event_objs[event_idx]
                event_conv = event_convs[event_idx]
                event_point = event_points[event_idx]
                event_que = event_ques[event_idx]
                ai_role, user_role = context_learner.get_roles(selected_event)
                
                # Display event information
                st.markdown(f"""
                <div class="event-info">
                    <div><strong>Event:</strong> {selected_event}</div>
                    <div><strong>Description:</strong> {event_desc}</div>
                    <div><strong>AI Role:</strong> {ai_role}</div>
                    <div><strong>User Role:</strong> {user_role}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Start Conversation", type="primary"):
                    chatbot = ChatBot(config)
                    persona_module = load_persona_prompts(selected_persona)
                    if persona_module:
                        start_new_conversation(selected_event, event_desc, event_obj, event_conv, event_point, event_que, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config, persona_module)
        
        # Conversation controls
        if st.session_state.conversation_started:
            st.subheader("Conversation Controls")
            
            if st.button("New Conversation"):
                st.session_state.conversation_started = False
                st.session_state.messages = []
                st.session_state.intro_completed = False
                st.session_state.training_started = False
                st.rerun()
            
            if st.button("Clear History"):
                st.session_state.messages = []
                st.rerun()
            
            # Display current event info
            st.info(f"**Current Event:** {st.session_state.current_event}\n\n**AI Role:** {st.session_state.ai_role}\n**User Role:** {st.session_state.user_role}")
            
            # Show conversation status
            if st.session_state.intro_completed and not st.session_state.training_started:
                st.warning("📖 **Status:** Introduction completed. Please type 'start' to begin training.")
            elif st.session_state.training_started:
                st.success("🎯 **Status:** Training in progress.")
            else:
                st.info("📝 **Status:** Introduction in progress.")
        
        # End Conversation and Export
        if st.sidebar.button("End Conversation & Export"):
            if st.session_state.get("messages"):
                # Save as JSON
                conversation_json = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.get('current_event','event')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
        # Display current event info at the top
        st.markdown(f"""
        <div class="event-info">
            <div><strong>Current Event:</strong> {st.session_state.current_event}</div>
            <div><strong>Description:</strong> {st.session_state.current_event_desc}</div>
            <div><strong>AI Role:</strong> {st.session_state.ai_role}</div>
            <div><strong>User Role:</strong> {st.session_state.user_role}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Show appropriate input prompt based on conversation state
        if st.session_state.intro_completed and not st.session_state.training_started:
            input_placeholder = "Type 'start' to begin training, or 'break' to end conversation..."
        elif st.session_state.training_started:
            input_placeholder = f"Type your response as {st.session_state.user_role}, or 'break' to end conversation..."
        else:
            input_placeholder = "Type your message here..."
        
        # Chat input
        user_input = st.chat_input(input_placeholder)
        
        if user_input:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process message and get response
            with st.spinner("Generating response..."):
                try:
                    response = process_user_message(user_input, st.session_state.chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
                    if response and response != "Conversation ended by user.":
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        # Welcome message
        st.info("👈 Please select a training scenario and persona from the sidebar to start a conversation.")
        
        # Display available events
        if context_learner:
            event_types, event_descs, event_objs, event_convs, event_points, event_ques = context_learner.get_types()
            
            st.subheader("Available Training Scenarios")
            for i, (event_type, event_desc) in enumerate(zip(event_types, event_descs)):
                with st.expander(f"{i+1}. {event_type}"):
                    st.write(f"**Description:** {event_desc}")
                    ai_role, user_role = context_learner.get_roles(event_type)
                    st.write(f"**AI Role:** {ai_role}")
                    st.write(f"**User Role:** {user_role}")

if __name__ == "__main__":
    main()
