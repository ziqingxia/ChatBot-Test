#!/usr/bin/env python3
"""
Standalone script to run scenario tests for chatbot comparison
"""

import os
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scenario_testing import ScenarioTester
from chatbot import ChatBot

def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_tests(config: Dict, num_runs: int = 5, test_levels: List[str] = None) -> Dict:
    """Run the scenario tests"""
    if test_levels is None:
        test_levels = ["Beginner", "Intermediate", "Advanced"]
    
    print(f"Starting scenario tests with {num_runs} runs per scenario")
    print(f"Testing levels: {', '.join(test_levels)}")
    
    # Initialize testers
    tester = ScenarioTester()
    
    # Initialize chatbots
    print("Initializing OpenAI chatbot...")
    openai_chatbot = ChatBot(config)
    
    print("Initializing BertChat...")
    try:
        tester = ScenarioTester()
        bert_chatbot = tester.create_bert_chatbot(config)
        if bert_chatbot:
            print("BertChat initialized successfully")
        else:
            print("BertChat initialization failed")
            bert_chatbot = None
    except Exception as e:
        print(f"BertChat initialization failed: {e}")
        bert_chatbot = None
    
    print("Initializing Pure BERT chatbot...")
    try:
        pure_bert_chatbot = tester.create_pure_bert_chatbot(config)
        if pure_bert_chatbot:
            print("Pure BERT chatbot initialized successfully")
        else:
            print("Pure BERT chatbot initialization failed")
            pure_bert_chatbot = None
    except Exception as e:
        print(f"Pure BERT chatbot initialization failed: {e}")
        pure_bert_chatbot = None
    
    # Filter scenarios based on selected levels
    filtered_scenarios = {level: scenarios for level, scenarios in tester.scenarios.items() 
                         if level in test_levels}
    
    print(f"Running tests for scenarios: {list(filtered_scenarios.keys())}")
    
    # Run tests
    results = tester.run_all_scenarios(
        openai_chatbot, 
        bert_chatbot, 
        pure_bert_chatbot,
        num_runs
    )
    
    return results

def save_results(results: Dict, output_dir: str = "test_results"):
    """Save test results to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Calculate and save summary
    tester = ScenarioTester()
    df = tester.calculate_scores(results)
    
    summary_file = os.path.join(output_dir, f"summary_results_{timestamp}.json")
    df.to_json(summary_file, orient='records', indent=2)
    
    # Save CSV summary
    csv_file = os.path.join(output_dir, f"summary_results_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Results saved to:")
    print(f"  Detailed: {detailed_file}")
    print(f"  Summary JSON: {summary_file}")
    print(f"  Summary CSV: {csv_file}")
    
    return detailed_file, summary_file, csv_file

def print_summary(results: Dict):
    """Print a summary of the test results"""
    tester = ScenarioTester()
    df = tester.calculate_scores(results)
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for level in df['Level'].unique():
        level_data = df[df['Level'] == level]
        print(f"\n{level} Level:")
        print("-" * 40)
        
        for _, row in level_data.iterrows():
            scenario_name = row['Scenario']
            openai_avg = row['OpenAI Total']
            bertchat_avg = row['BertChat Total']
            pure_bert_avg = row['Pure BERT Total']
            
            print(f"{scenario_name}:")
            print(f"  OpenAI: {openai_avg}/10")
            print(f"  BertChat: {bertchat_avg}/10")
            print(f"  Pure BERT: {pure_bert_avg}/10")
    
    # Overall summary
    overall_openai = df['OpenAI Total'].mean()
    overall_bertchat = df['BertChat Total'].mean()
    overall_pure_bert = df['Pure BERT Total'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  OpenAI Average: {overall_openai:.1f}/10")
    print(f"  BertChat Average: {overall_bertchat:.1f}/10")
    print(f"  Pure BERT Average: {overall_pure_bert:.1f}/10")
    
    # Find winner
    scores = [
        ("OpenAI", overall_openai),
        ("BertChat", overall_bertchat),
        ("Pure BERT", overall_pure_bert)
    ]
    winner = max(scores, key=lambda x: x[1])
    print(f"  🏆 Winner: {winner[0]} ({winner[1]:.1f}/10)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run chatbot scenario tests")
    parser.add_argument("--runs", type=int, default=5, help="Number of test runs per scenario")
    parser.add_argument("--levels", nargs="+", default=["Beginner", "Intermediate", "Advanced"], 
                       help="Test levels to run")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output", default="test_results", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running tests")
        sys.exit(1)
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Run tests
        results = run_tests(config, args.runs, args.levels)
        
        # Print summary
        print_summary(results)
        
        # Save results
        if not args.no_save:
            save_results(results, args.output)
        
        print("\nTest run completed successfully!")
        
    except Exception as e:
        print(f"Error during test run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 