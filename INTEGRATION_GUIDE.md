# Integration Guide: Adding Testing System to Main Streamlit App

This guide shows how to integrate the scenario testing system into your existing Streamlit application.

## Quick Integration

### Step 1: Add Testing Tab to Main App

Add this code to your `streamlit_app.py` to include a testing tab:

```python
# Add this import at the top of your file
from scenario_testing import ScenarioTester

# Add this after your existing page configuration
def testing_page():
    """Testing page for scenario evaluation"""
    st.title("🧪 Chatbot Testing System")
    
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
    
    # Run tests button
    if st.button("🚀 Run All Tests", type="primary"):
        with st.spinner("Initializing chatbots..."):
            # Initialize OpenAI chatbot
            openai_chatbot = ChatBot(config)
            
            # Initialize BERT chatbot if available
            bert_chatbot = None
            try:
                from bert_chatbot import BertChatBot
                bert_chatbot = BertChatBot(config)
                st.success("BERT chatbot initialized successfully")
            except Exception as e:
                st.warning(f"BERT chatbot initialization failed: {e}")
        
        # Run tests
        with st.spinner("Running tests..."):
            results = tester.run_all_scenarios(
                openai_chatbot, 
                bert_chatbot, 
                num_runs
            )
        
        # Display results
        df = tester.calculate_scores(results)
        tester.display_results_table(df)

# Modify your main() function to include the testing page
def main():
    # Add page selection
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Chat", "Testing", "Settings"]
    )
    
    if page == "Chat":
        # Your existing chat functionality
        run_benchmark_app()
    elif page == "Testing":
        testing_page()
    elif page == "Settings":
        # Your settings page
        pass
```

### Step 2: Add Testing Button to Existing Chat Interface

Alternatively, add a testing button to your existing chat interface:

```python
# Add this to your existing chat interface
if st.sidebar.button("🧪 Run Scenario Tests"):
    st.session_state['show_testing'] = True

# In your main chat area, add this condition
if st.session_state.get('show_testing', False):
    testing_page()
    if st.button("← Back to Chat"):
        st.session_state['show_testing'] = False
        st.rerun()
```

## Advanced Integration

### Custom Testing Interface

Create a more integrated testing interface:

```python
def integrated_testing_interface():
    """Integrated testing interface with your existing chatbot"""
    
    st.title("🧪 Integrated Testing System")
    
    # Use your existing chatbot instances
    if 'chatbot' in st.session_state:
        openai_chatbot = st.session_state['chatbot']
    else:
        openai_chatbot = ChatBot(config)
        st.session_state['chatbot'] = openai_chatbot
    
    # Initialize BERT chatbot
    if 'bert_chatbot' not in st.session_state:
        try:
            from bert_chatbot import BertChatBot
            st.session_state['bert_chatbot'] = BertChatBot(config)
        except Exception as e:
            st.session_state['bert_chatbot'] = None
    
    bert_chatbot = st.session_state['bert_chatbot']
    
    # Test configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_runs = st.number_input("Test runs", 1, 20, 5)
    
    with col2:
        test_level = st.selectbox("Test Level", ["Beginner", "Intermediate", "Advanced"])
    
    with col3:
        if st.button("Run Tests", type="primary"):
            run_integrated_tests(openai_chatbot, bert_chatbot, test_level, num_runs)

def run_integrated_tests(openai_chatbot, bert_chatbot, level, num_runs):
    """Run tests with progress tracking"""
    
    tester = ScenarioTester()
    
    # Filter scenarios for selected level
    scenarios = {level: tester.scenarios[level]}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    total_scenarios = len(scenarios[level])
    current_scenario = 0
    
    for scenario_id, scenario_data in scenarios[level].items():
        status_text.text(f"Testing {level} - {scenario_id}...")
        
        # Run OpenAI tests
        openai_results = []
        for run in range(num_runs):
            try:
                run_results = tester.run_scenario(openai_chatbot, scenario_data, scenario_id)
                openai_results.append(run_results)
            except Exception as e:
                st.error(f"Error in OpenAI run {run + 1}: {str(e)}")
        
        # Run BERT tests if available
        bert_results = []
        if bert_chatbot:
            for run in range(num_runs):
                try:
                    run_results = tester.run_scenario(bert_chatbot, scenario_data, scenario_id)
                    bert_results.append(run_results)
                except Exception as e:
                    st.error(f"Error in BERT run {run + 1}: {str(e)}")
        
        results[scenario_id] = {
            "name": scenario_data["name"],
            "openai_runs": openai_results,
            "bert_runs": bert_results
        }
        
        current_scenario += 1
        progress_bar.progress(current_scenario / total_scenarios)
    
    # Display results
    display_integrated_results(results, level)

def display_integrated_results(results, level):
    """Display results in an integrated format"""
    
    st.success("✅ Tests completed!")
    
    # Create results table
    data = []
    for scenario_id, scenario_data in results.items():
        # Calculate OpenAI scores
        openai_scores = []
        for run in scenario_data["openai_runs"]:
            run_scores = [turn["score"] for turn in run if turn["score"] is not None]
            openai_scores.extend(run_scores)
        
        openai_avg = sum(openai_scores) / len(openai_scores) if openai_scores else 0
        
        # Calculate BERT scores
        bert_scores = []
        for run in scenario_data["bert_runs"]:
            run_scores = [turn["score"] for turn in run if turn["score"] is not None]
            bert_scores.extend(run_scores)
        
        bert_avg = sum(bert_scores) / len(bert_scores) if bert_scores else 0
        
        data.append({
            "Scenario": f"{scenario_id} - {scenario_data['name']}",
            "OpenAI": f"{openai_avg:.1f}/10",
            "BERT": f"{bert_avg:.1f}/10" if bert_scores else "N/A"
        })
    
    # Display table
    df = pd.DataFrame(data)
    st.table(df)
    
    # Download results
    if st.button("📥 Download Results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"integrated_test_results_{level}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(results, indent=2, default=str),
            file_name=results_file,
            mime="application/json"
        )
```

### Session State Management

Add proper session state management for testing:

```python
def initialize_testing_session_state():
    """Initialize session state for testing"""
    if 'testing_results' not in st.session_state:
        st.session_state['testing_results'] = {}
    
    if 'testing_in_progress' not in st.session_state:
        st.session_state['testing_in_progress'] = False
    
    if 'current_test_level' not in st.session_state:
        st.session_state['current_test_level'] = None

# Add this to your main function
def main():
    initialize_testing_session_state()
    # ... rest of your main function
```

## Integration with Existing Features

### Combine with Database Management

```python
def testing_with_database():
    """Testing interface that uses your existing database"""
    
    st.title("🧪 Database-Aware Testing")
    
    # Use your existing RAG system
    if 'rag_database' in st.session_state:
        rag_database = st.session_state['rag_database']
    else:
        rag_database = RAGKnowledgeBase(config)
        st.session_state['rag_database'] = rag_database
    
    # Test scenarios with database context
    tester = ScenarioTester()
    
    # Modify the test method to include database context
    def enhanced_test_scenario(chatbot, scenario_data, scenario_name):
        results = []
        conversation = scenario_data["conversation"]
        
        chatbot.clear_history()
        
        for i, turn in enumerate(conversation):
            if turn["role"] == "user":
                user_message = turn["message"]
                
                # Get database context
                data_content = rag_database.search(user_message)
                
                try:
                    response = chatbot.test(user_message, data_content, None)
                    results.append({
                        "turn": i + 1,
                        "user_message": user_message,
                        "chatbot_response": response,
                        "database_context": data_content,
                        "expected_response": None,
                        "score": None
                    })
                except Exception as e:
                    results.append({
                        "turn": i + 1,
                        "user_message": user_message,
                        "chatbot_response": f"ERROR: {str(e)}",
                        "database_context": data_content,
                        "expected_response": None,
                        "score": None
                    })
            
            elif turn["role"] == "expected":
                expected_response = turn["message"]
                
                if results and results[-1]["chatbot_response"] and not results[-1]["chatbot_response"].startswith("ERROR"):
                    actual_response = results[-1]["chatbot_response"]
                    score = tester.evaluate_response(expected_response, actual_response)
                    
                    results[-1]["expected_response"] = expected_response
                    results[-1]["score"] = score
        
        return results
    
    # Use the enhanced testing
    if st.button("Run Database-Aware Tests"):
        # ... implementation
        pass
```

### Integration with History System

```python
def testing_with_history():
    """Testing that integrates with your history system"""
    
    # Use your existing history management
    history_path = config['CHATBOT']['HISTORY_PATH']
    
    # Save test results to history
    def save_test_to_history(test_results, test_name):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        history_file = os.path.join(history_path, f"{timestamp}_{test_name}.json")
        
        with open(history_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        return history_file
    
    # Load previous test results
    def load_test_history():
        history_files = glob.glob(os.path.join(history_path, "*_test_*.json"))
        return sorted(history_files, reverse=True)
    
    # Display test history
    st.subheader("Previous Test Results")
    history_files = load_test_history()
    
    if history_files:
        selected_history = st.selectbox("Select previous test", history_files)
        
        if selected_history:
            with open(selected_history, 'r') as f:
                historical_results = json.load(f)
            
            st.json(historical_results)
    else:
        st.info("No previous test results found")
```

## Best Practices

1. **Error Handling**: Always wrap testing code in try-catch blocks
2. **Progress Tracking**: Use Streamlit's progress bars for long-running tests
3. **Session State**: Use session state to persist testing data across reruns
4. **Resource Management**: Clear chatbot history between tests
5. **User Feedback**: Provide clear status messages and error explanations

## Troubleshooting Integration

### Common Issues

1. **Import Errors**: Ensure all testing modules are in the same directory
2. **Session State Conflicts**: Use unique keys for testing session state
3. **Memory Issues**: Clear chatbot history and session state between tests
4. **API Rate Limits**: Add delays between API calls in testing loops

### Debug Mode

```python
# Add debug mode for testing
if st.sidebar.checkbox("Debug Mode"):
    st.write("Session State:", st.session_state)
    st.write("Config:", config)
```

This integration guide provides multiple approaches to add the testing system to your existing Streamlit app, from simple tab-based integration to advanced database-aware testing. 