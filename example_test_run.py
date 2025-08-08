#!/usr/bin/env python3
"""
Example script demonstrating how to use the scenario testing system
"""

import os
import sys
import yaml
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scenario_testing import ScenarioTester
from chatbot import ChatBot

def main():
    """Example usage of the scenario testing system"""
    
    print("🚀 Chatbot Scenario Testing Example")
    print("=" * 50)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running this example")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load configuration
    try:
        with open('configs/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Initialize testers
    tester = ScenarioTester()
    print("✅ Scenario tester initialized")
    
    # Initialize chatbots
    print("\n🤖 Initializing chatbots...")
    
    try:
        openai_chatbot = ChatBot(config)
        print("✅ OpenAI chatbot initialized")
    except Exception as e:
        print(f"❌ Error initializing OpenAI chatbot: {e}")
        return
    
    try:
        tester = ScenarioTester()
        bert_chatbot = tester.create_bert_chatbot(config)
        if bert_chatbot:
            print("✅ BertChat initialized")
        else:
            print("⚠️  Warning: BertChat initialization failed")
            bert_chatbot = None
    except Exception as e:
        print(f"⚠️  Warning: BertChat initialization failed: {e}")
        print("Continuing with OpenAI only...")
        bert_chatbot = None
    
    try:
        pure_bert_chatbot = tester.create_pure_bert_chatbot(config)
        if pure_bert_chatbot:
            print("✅ Pure BERT chatbot initialized")
        else:
            print("⚠️  Warning: Pure BERT chatbot initialization failed")
            pure_bert_chatbot = None
    except Exception as e:
        print(f"⚠️  Warning: Pure BERT chatbot initialization failed: {e}")
        print("Continuing with available models...")
        pure_bert_chatbot = None
    
    # Run a quick test with just one scenario
    print("\n🧪 Running quick test...")
    
    # Get the first scenario from each level
    test_scenarios = {}
    for level, scenarios in tester.scenarios.items():
        first_scenario_id = list(scenarios.keys())[0]
        test_scenarios[level] = {first_scenario_id: scenarios[first_scenario_id]}
    
    # Run tests with just 2 iterations for quick demo
    results = tester.run_all_scenarios(
        openai_chatbot, 
        bert_chatbot, 
        pure_bert_chatbot,
        num_runs=2  # Quick test with 2 runs
    )
    
    # Calculate and display results
    print("\n📊 Test Results:")
    print("=" * 50)
    
    df = tester.calculate_scores(results)
    
    for level in df['Level'].unique():
        level_data = df[df['Level'] == level]
        print(f"\n{level} Level:")
        print("-" * 30)
        
        for _, row in level_data.iterrows():
            scenario_name = row['Scenario']
            openai_avg = row['OpenAI Total']
            bert_avg = row['BERT Total']
            
            print(f"{scenario_name}:")
            print(f"  OpenAI: {openai_avg}/10")
            if bert_chatbot:
                print(f"  BERT:   {bert_avg}/10")
    
    # Overall summary
    overall_openai = df['OpenAI Total'].mean()
    print(f"\nOverall Performance:")
    print(f"  OpenAI Average: {overall_openai:.1f}/10")
    
    if bert_chatbot:
        overall_bert = df['BERT Total'].mean()
        print(f"  BERT Average:   {overall_bert:.1f}/10")
        
        if overall_openai > overall_bert:
            print(f"  🏆 Winner: OpenAI (+{overall_openai - overall_bert:.1f})")
        elif overall_bert > overall_openai:
            print(f"  🏆 Winner: BERT (+{overall_bert - overall_openai:.1f})")
        else:
            print(f"  🤝 Tie")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"example_results_{timestamp}.json"
    
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")
    
    print("\n✅ Example test run completed!")
    print("\nTo run full tests, use:")
    print("  python run_scenario_tests.py --runs 5")
    print("\nOr launch the Streamlit interface:")
    print("  streamlit run scenario_testing.py")

if __name__ == "__main__":
    main() 