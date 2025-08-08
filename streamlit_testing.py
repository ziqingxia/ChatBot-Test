import streamlit as st
import os
import json
import yaml
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import testing framework
from test_scenarios import ScenarioTester, TestScenario, ModelType

# Page configuration
st.set_page_config(
    page_title="Chatbot Testing Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .scenario-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    if 'testing_in_progress' not in st.session_state:
        st.session_state.testing_in_progress = False
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    if 'total_tests' not in st.session_state:
        st.session_state.total_tests = 0
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    if 'tester_initialized' not in st.session_state:
        st.session_state.tester_initialized = False

def load_config():
    """Load configuration from YAML file."""
    try:
        with open("configs/config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

def check_environment():
    """Check if the environment is properly set up."""
    issues = []
    
    # Check config file
    if not os.path.exists("configs/config.yaml"):
        issues.append("Configuration file not found (configs/config.yaml)")
    
    # Check database directories
    if not os.path.exists("database"):
        issues.append("Database directory not found")
    
    if not os.path.exists("dictionary"):
        issues.append("Dictionary directory not found")
    
    return issues

def set_api_key(api_key):
    """Set the OpenAI API key in environment."""
    os.environ['OPENAI_API_KEY'] = api_key

def run_tests_with_progress(tester, selected_scenarios, selected_models, num_runs):
    """Run tests with progress tracking."""
    st.session_state.testing_in_progress = True
    st.session_state.test_results = []
    
    total_tests = len(selected_scenarios) * len(selected_models) * num_runs
    st.session_state.total_tests = total_tests
    st.session_state.current_progress = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for scenario in selected_scenarios:
            for model_type in selected_models:
                for run in range(1, num_runs + 1):
                    # Update progress
                    st.session_state.current_progress += 1
                    progress = st.session_state.current_progress / total_tests
                    progress_bar.progress(progress)
                    
                    status_text.text(f"Testing {scenario.name} - {model_type.value} (Run {run}/{num_runs})")
                    
                    # Run test
                    if model_type == ModelType.OPENAI:
                        result = tester.run_openai_test(scenario, run)
                    else:  # BERT
                        result = tester.run_bert_test(scenario, run)
                    
                    if result:
                        st.session_state.test_results.append(result)
                    
                    # Small delay to show progress
                    time.sleep(0.5)
        
        status_text.text("✅ Testing completed!")
        progress_bar.progress(1.0)
        
    except Exception as e:
        st.error(f"Error during testing: {e}")
    finally:
        st.session_state.testing_in_progress = False

def create_comparison_table(results):
    """Create a comparison table from results."""
    if not results:
        return pd.DataFrame()
    
    # Group results by scenario
    scenario_data = {}
    for result in results:
        if result.scenario_name not in scenario_data:
            scenario_data[result.scenario_name] = {"OpenAI": [], "BERT": []}
        
        if result.model_type == ModelType.OPENAI:
            scenario_data[result.scenario_name]["OpenAI"].append(result.total_score)
        else:
            scenario_data[result.scenario_name]["BERT"].append(result.total_score)
    
    # Create DataFrame
    table_data = []
    for scenario_name, scores in scenario_data.items():
        openai_scores = scores["OpenAI"]
        bert_scores = scores["BERT"]
        
        openai_avg = sum(openai_scores) / len(openai_scores) if openai_scores else 0
        bert_avg = sum(bert_scores) / len(bert_scores) if bert_scores else 0
        
        table_data.append({
            "Scenario": scenario_name,
            "OpenAI Scores": " | ".join([f"{s:.1f}" for s in openai_scores]),
            "OpenAI Average": f"{openai_avg:.1f}",
            "BERT Scores": " | ".join([f"{s:.1f}" for s in bert_scores]) if bert_scores else "N/A",
            "BERT Average": f"{bert_avg:.1f}" if bert_scores else "N/A"
        })
    
    return pd.DataFrame(table_data)

def create_score_chart(results):
    """Create a score comparison chart."""
    if not results:
        return None
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        plot_data.append({
            "Scenario": result.scenario_name,
            "Model": result.model_type.value,
            "Run": result.run_number,
            "Score": result.total_score
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Score Distribution by Model", "Average Scores by Scenario"),
        specs=[[{"type": "box"}, {"type": "bar"}]]
    )
    
    # Box plot for score distribution
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]["Score"]
        fig.add_trace(
            go.Box(y=model_data, name=model, boxpoints="all"),
            row=1, col=1
        )
    
    # Bar plot for average scores by scenario
    avg_scores = df.groupby(["Scenario", "Model"])["Score"].mean().reset_index()
    
    for model in avg_scores["Model"].unique():
        model_data = avg_scores[avg_scores["Model"] == model]
        fig.add_trace(
            go.Bar(
                x=model_data["Scenario"],
                y=model_data["Score"],
                name=f"{model} (Avg)",
                text=[f"{s:.1f}" for s in model_data["Score"]],
                textposition="auto"
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Test Results Analysis"
    )
    
    return fig

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🧪 Chatbot Testing Framework</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Environment check
        st.markdown("#### Environment Check")
        issues = check_environment()
        if issues:
            st.error("Environment issues detected:")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ Environment OK")
        
        # API Key Status
        api_key_status = "✅ Set" if os.getenv('OPENAI_API_KEY') else "❌ Not Set"
        st.info(f"**OpenAI API Key:** {api_key_status}")
        
        # Test configuration
        st.markdown("#### Test Settings")
        
        # API Key Input
        st.markdown("#### 🔑 OpenAI API Key")
        
        # Check if API key is already set
        current_api_key = os.getenv('OPENAI_API_KEY', '')
        
        if current_api_key:
            st.success("✅ API key already set")
            if st.button("🔄 Change API Key"):
                st.session_state.api_key_set = False
                st.rerun()
        else:
            api_key = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your OpenAI API key is required to run tests with OpenAI models"
            )
            
            if api_key:
                if api_key.startswith("sk-") and len(api_key) > 20:
                    set_api_key(api_key)
                    st.session_state.api_key_set = True
                    st.success("✅ API key set successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format. Please enter a valid OpenAI API key.")
        
        # Initialize tester variable
        tester = None
        
        # Scenario selection
        st.markdown("#### Select Scenarios")
        
        # Try to load scenarios if tester is available
        all_scenarios = []
        try:
            if not st.session_state.tester_initialized:
                tester = ScenarioTester()
                st.session_state.tester_initialized = True
                st.session_state.tester = tester
                st.success("✅ Testing framework loaded")
            else:
                tester = st.session_state.tester
                st.success("✅ Testing framework ready")
            
            all_scenarios = tester.scenarios
        except Exception as e:
            st.error(f"❌ Failed to load testing framework: {e}")
            if "Invalid OpenAI API key" in str(e):
                st.error("Please check your API key and try again.")
            # Continue with empty scenarios list
            all_scenarios = []
        
        # Group scenarios by level
        beginner_scenarios = [s for s in all_scenarios if s.level == "Beginner"]
        intermediate_scenarios = [s for s in all_scenarios if s.level == "Intermediate"]
        advanced_scenarios = [s for s in all_scenarios if s.level == "Advanced"]
        
        selected_scenarios = []
        
        if not all_scenarios:
            st.warning("⚠️ No scenarios available. Please check your API key and try again.")
        else:
            with st.expander("Beginner Level", expanded=True):
                for scenario in beginner_scenarios:
                    if st.checkbox(f"{scenario.name} - {scenario.level}", key=f"beginner_{scenario.name}"):
                        selected_scenarios.append(scenario)
            
            with st.expander("Intermediate Level", expanded=True):
                for scenario in intermediate_scenarios:
                    if st.checkbox(f"{scenario.name} - {scenario.level}", key=f"intermediate_{scenario.name}"):
                        selected_scenarios.append(scenario)
            
            with st.expander("Advanced Level", expanded=True):
                for scenario in advanced_scenarios:
                    if st.checkbox(f"{scenario.name} - {scenario.level}", key=f"advanced_{scenario.name}"):
                        selected_scenarios.append(scenario)
        
        # Model selection
        st.markdown("#### Select Models")
        selected_models = []
        
        if st.checkbox("OpenAI (GPT-4)", value=True, key="openai_model"):
            selected_models.append(ModelType.OPENAI)
        
        if st.checkbox("BERT", value=True, key="bert_model"):
            selected_models.append(ModelType.BERT)
        
        # Load tester if needed and not already loaded
        if selected_models and tester is None:
            # Check if we can load tester based on selected models
            can_load_tester = (
                ModelType.BERT in selected_models or 
                (ModelType.OPENAI in selected_models and os.getenv('OPENAI_API_KEY'))
            )
            
            if can_load_tester:
                try:
                    tester = ScenarioTester()
                    st.session_state.tester_initialized = True
                    st.session_state.tester = tester
                    st.success("✅ Testing framework loaded")
                    all_scenarios = tester.scenarios
                except Exception as e:
                    st.error(f"❌ Failed to load testing framework: {e}")
                    if "Invalid OpenAI API key" in str(e):
                        st.error("Please check your API key and try again.")
            else:
                st.warning("⚠️ Please set your OpenAI API key to use OpenAI models")
        
        # Number of runs
        num_runs = st.slider("Number of test runs per scenario", 1, 10, 5)
        
        # Run tests button
        st.markdown("#### Run Tests")
        
        # Check if we can run tests
        can_run_tests = (
            selected_scenarios and 
            selected_models and 
            tester is not None and
            (ModelType.BERT in selected_models or os.getenv('OPENAI_API_KEY'))
        )
        
        if st.button("🚀 Start Testing", type="primary", disabled=st.session_state.testing_in_progress or not can_run_tests):
            if not selected_scenarios:
                st.error("Please select at least one scenario")
                return
            if not selected_models:
                st.error("Please select at least one model")
                return
            if tester is None:
                st.error("Testing framework not loaded. Please check your API key and try again.")
                return
            if ModelType.OPENAI in selected_models and not os.getenv('OPENAI_API_KEY'):
                st.error("OpenAI API key required for OpenAI model testing")
                return
            
            # Run tests
            run_tests_with_progress(tester, selected_scenarios, selected_models, num_runs)
            st.rerun()
        
        # Show why tests can't run
        if not can_run_tests:
            if not selected_scenarios:
                st.info("ℹ️ Select at least one scenario to run tests")
            elif not selected_models:
                st.info("ℹ️ Select at least one model to run tests")
            elif tester is None:
                st.info("ℹ️ Testing framework not loaded")
            elif ModelType.OPENAI in selected_models and not os.getenv('OPENAI_API_KEY'):
                st.info("ℹ️ Set your OpenAI API key to test OpenAI models")
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.test_results = []
            st.rerun()
    
    # Main content area
    if st.session_state.testing_in_progress:
        st.info("🔄 Testing in progress... Please wait.")
        return
    
    # Display results
    if st.session_state.test_results:
        st.markdown("## 📊 Test Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_tests = len(st.session_state.test_results)
        openai_results = [r for r in st.session_state.test_results if r.model_type == ModelType.OPENAI]
        bert_results = [r for r in st.session_state.test_results if r.model_type == ModelType.BERT]
        
        with col1:
            st.metric("Total Tests", total_tests)
        
        with col2:
            openai_avg = sum(r.total_score for r in openai_results) / len(openai_results) if openai_results else 0
            st.metric("OpenAI Average", f"{openai_avg:.2f}")
        
        with col3:
            bert_avg = sum(r.total_score for r in bert_results) / len(bert_results) if bert_results else 0
            st.metric("BERT Average", f"{bert_avg:.2f}")
        
        with col4:
            overall_avg = sum(r.total_score for r in st.session_state.test_results) / len(st.session_state.test_results)
            st.metric("Overall Average", f"{overall_avg:.2f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Comparison Table", "📈 Charts", "📝 Detailed Results", "💾 Export"])
        
        with tab1:
            st.markdown("### Comparison Table")
            comparison_df = create_comparison_table(st.session_state.test_results)
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("No results to display")
        
        with tab2:
            st.markdown("### Score Analysis")
            chart = create_score_chart(st.session_state.test_results)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No results to display")
        
        with tab3:
            st.markdown("### Detailed Results")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                selected_scenario = st.selectbox(
                    "Filter by Scenario",
                    ["All"] + list(set(r.scenario_name for r in st.session_state.test_results))
                )
            
            with col2:
                selected_model = st.selectbox(
                    "Filter by Model",
                    ["All"] + list(set(r.model_type.value for r in st.session_state.test_results))
                )
            
            # Filter results
            filtered_results = st.session_state.test_results
            if selected_scenario != "All":
                filtered_results = [r for r in filtered_results if r.scenario_name == selected_scenario]
            if selected_model != "All":
                filtered_results = [r for r in filtered_results if r.model_type.value == selected_model]
            
            # Display detailed results
            for result in filtered_results:
                with st.expander(f"{result.scenario_name} - {result.model_type.value} (Run {result.run_number})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Score", f"{result.total_score:.2f}/5.0")
                        st.write("**Individual Scores:**")
                        for i, score in enumerate(result.scores):
                            st.write(f"Response {i+1}: {score:.2f}")
                    
                    with col2:
                        st.write("**Conversation Log:**")
                        for turn in result.conversation_log:
                            role_emoji = "👤" if turn["role"] == "user" else "🤖"
                            st.write(f"{role_emoji} **{turn['role'].title()}:** {turn['content']}")
        
        with tab4:
            st.markdown("### Export Results")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export as JSON"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_data = []
                    
                    for result in st.session_state.test_results:
                        json_data.append({
                            "scenario_name": result.scenario_name,
                            "model_type": result.model_type.value,
                            "run_number": result.run_number,
                            "scores": result.scores,
                            "total_score": result.total_score,
                            "responses": result.responses,
                            "conversation_log": result.conversation_log
                        })
                    
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(json_data, indent=2, ensure_ascii=False),
                        file_name=f"test_results_{timestamp}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📊 Export as CSV"):
                    comparison_df = create_comparison_table(st.session_state.test_results)
                    if not comparison_df.empty:
                        csv_data = comparison_df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"comparison_table_{timestamp}.csv",
                            mime="text/csv"
                        )
    
    else:
        # Welcome message
        st.markdown("## 👋 Welcome to the Chatbot Testing Framework")
        
        # Check if API key is set
        api_key_set = bool(os.getenv('OPENAI_API_KEY'))
        
        if not api_key_set:
            st.warning("""
            ### 🔑 API Key Required
            
            **To use OpenAI models**, you need to set your OpenAI API key in the sidebar.
            
            **To use BERT models only**, you can proceed without an API key.
            
            **Getting an OpenAI API Key:**
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign in or create an account
            3. Click "Create new secret key"
            4. Copy the key (starts with `sk-`)
            5. Paste it in the sidebar
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What this framework does:
            
            - **Tests both OpenAI and BERT chatbots** across various scenarios
            - **Evaluates radio communication protocols** at different difficulty levels
            - **Runs multiple iterations** to ensure consistency
            - **Generates detailed reports** with scoring and analysis
            - **Provides interactive visualizations** of results
            
            ### 📋 Available Scenarios:
            
            **Beginner Level:**
            - Scenario 2.1: Basic radio check with consistent naming
            - Scenario 2.2: Radio check with different phrasing
            - Scenario 2.3: Control set with standard protocol
            
            **Intermediate Level:**
            - Scenario 6.1: TOA surrender with full protocol
            - Scenario 6.2: TOA surrender with different phrasing
            - Scenario 6.3: Control set with standard TOA protocol
            
            **Advanced Level:**
            - Scenario 14.1: Emergency rail crack response
            - Scenario 14.2: Emergency response with different phrasing
            - Scenario 14.3: Emergency response with full location details
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Getting Started:
            
            1. **Set your API key** in the sidebar (if using OpenAI)
            2. **Check environment status** in the sidebar
            3. **Select scenarios** you want to test
            4. **Choose models** (OpenAI and/or BERT)
            5. **Set number of runs** per scenario
            6. **Click "Start Testing"** to begin
            
            ### 📊 Understanding Results:
            
            **Scoring System (0-5):**
            - **5.0**: Perfect protocol adherence
            - **4.0-4.9**: Excellent with minor variations
            - **3.0-3.9**: Good protocol understanding
            - **2.0-2.9**: Fair basic knowledge
            - **1.0-1.9**: Poor protocol knowledge
            - **0.0-0.9**: Incorrect responses
            
            ### 💡 Tips:
            
            - Start with a few scenarios to test the framework
            - Use BERT-only testing if you don't have an API key
            - Export results for further analysis
            - Check detailed logs for specific issues
            """)

if __name__ == "__main__":
    main() 