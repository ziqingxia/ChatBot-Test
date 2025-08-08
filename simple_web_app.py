import streamlit as st
import os
import yaml
import json
import torch
from datetime import datetime
import re

from chatbot import ChatBot, InvalidAPIKeyError
from database_web import RAGKnowledgeBase
from fewshot import InContextLearner

# Page configuration
st.set_page_config(
    page_title="Safety ChatBot System",
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
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
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

def start_new_conversation(event_name, event_desc, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config):
    """Start a new conversation"""
    st.session_state.current_event = event_name
    st.session_state.current_event_desc = event_desc
    st.session_state.ai_role = ai_role
    st.session_state.user_role = user_role
    st.session_state.chatbot = chatbot
    st.session_state.conversation_started = True
    st.session_state.messages = []
    
    # Set chat type
    chatbot.set_chat_type(chat_type="conversation")
    
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

def clean_response(text):
    """Clean response text by removing HTML tags and code blocks"""
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
        
        return clean_response(response)
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
    st.markdown('<h1 class="main-header">🤖 Safety ChatBot System</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Event selection
        if not st.session_state.conversation_started:
            event_types, event_descs = context_learner.get_types()
            
            # Event selection dropdown
            selected_event = st.selectbox(
                "Choose an event type:",
                event_types,
                format_func=lambda x: f"{x}",
                help="Select the type of safety event to practice"
            )
            
            if selected_event:
                event_desc = event_descs[event_types.index(selected_event)]
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
                    start_new_conversation(selected_event, event_desc, ai_role, user_role, chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
        
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
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process message and get response
            with st.spinner("Generating response..."):
                try:
                    response = process_user_message(user_input, st.session_state.chatbot, rag_database, rag_dictionary, knowledge_phrases, config)
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

if __name__ == "__main__":
    main() 