# Chatbot Scenario Testing System

This system provides comprehensive testing capabilities for comparing OpenAI and BERT-based chatbot implementations across predefined railway communication scenarios.

## Overview

The testing system evaluates chatbot performance using:
- **Predefined scenarios** across three difficulty levels (Beginner, Intermediate, Advanced)
- **Automated scoring** based on response accuracy and protocol adherence
- **Multiple test runs** to ensure consistency and reliability
- **Comparative analysis** between OpenAI and BERT implementations

## Test Scenarios

### Beginner Level
- **Scenario 2.1**: Name Recognition and Consistency
- **Scenario 2.2**: Different Phrasing Test  
- **Scenario 2.3**: Control Set

### Intermediate Level
- **Scenario 6.1**: Naming Recognition and Consistency
- **Scenario 6.2**: Different Phrasing Test
- **Scenario 6.3**: Control Set

### Advanced Level
- **Scenario 14.1**: Protocol Adherence
- **Scenario 14.2**: Different Phrasing Test
- **Scenario 14.3**: Protocol Adherence (Detailed)

## Scoring System

Each chatbot response is scored on a scale of 1-10 based on:
- **Exact phrase matching** (10 points for perfect match)
- **Key phrase recognition** (partial credit for matching important terms)
- **Protocol adherence** (following railway communication standards)
- **Contextual appropriateness** (response fits the conversation flow)

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Verify BERT dependencies**:
   ```bash
   python -c "import sentence_transformers; print('BERT ready')"
   ```

## Usage

### Method 1: Standalone Script (Recommended)

Run tests from command line:

```bash
# Run all scenarios with 5 iterations each
python run_scenario_tests.py

# Run specific levels
python run_scenario_tests.py --levels Beginner Intermediate

# Run with custom number of iterations
python run_scenario_tests.py --runs 10

# Save results to custom directory
python run_scenario_tests.py --output my_test_results
```

### Method 2: Streamlit Interface

Launch the testing interface:

```bash
streamlit run scenario_testing.py
```

### Method 3: Integration with Main App

The testing system can be integrated into your main Streamlit app by importing the `ScenarioTester` class.

## Output Format

### Results Table Format

| Level | Scenario | OpenAI Scores | OpenAI Total | BERT Scores | BERT Total |
|-------|----------|---------------|--------------|-------------|------------|
| Beginner | 2.1 - Name Recognition | 3,4,3,5,4 | 3.8 | 4,3,4,3,4 | 3.6 |
| Beginner | 2.2 - Different Phrasing | 6,8,7,6,7 | 6.8 | 5,6,7,6,5 | 5.8 |
| ... | ... | ... | ... | ... | ... |

### Output Files

The system generates three types of output files:

1. **Detailed Results** (`detailed_results_TIMESTAMP.json`):
   - Complete conversation logs
   - Individual response evaluations
   - Error logs and debugging information

2. **Summary JSON** (`summary_results_TIMESTAMP.json`):
   - Aggregated scores by scenario
   - Performance comparisons
   - Statistical summaries

3. **Summary CSV** (`summary_results_TIMESTAMP.csv`):
   - Tabular format for analysis
   - Compatible with Excel/Google Sheets
   - Easy to import into other tools

## Configuration

### Config File (`configs/config.yaml`)

Key settings for testing:

```yaml
SEARCH:
  TOPK: 5                  # Number of context items to retrieve
  THRESHOLD: 0.3           # Similarity threshold for context matching
  DISPLAY_LENGTH: 100      # Context display length

CHATBOT:
  HISTORY_PATH: "./history"  # Where to save conversation logs
```

### Customizing Scenarios

To add new test scenarios, modify the `_load_test_scenarios()` method in `scenario_testing.py`:

```python
def _load_test_scenarios(self) -> Dict[str, List[Dict]]:
    return {
        "Your Level": {
            "scenario_id": {
                "name": "Scenario Description",
                "conversation": [
                    {"role": "user", "message": "User input"},
                    {"role": "expected", "message": "Expected response"},
                    # ... more turns
                ]
            }
        }
    }
```

## Troubleshooting

### Common Issues

1. **BERT Import Errors**:
   ```bash
   pip install sentence-transformers scikit-learn
   ```

2. **OpenAI API Errors**:
   - Verify API key is set correctly
   - Check API key has sufficient credits
   - Ensure network connectivity

3. **Memory Issues**:
   - Reduce `TOPK` in config
   - Use smaller BERT model
   - Run fewer concurrent tests

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### For Large-Scale Testing

1. **Batch Processing**:
   ```bash
   # Run tests in parallel (if you have multiple API keys)
   python run_scenario_tests.py --runs 1 --levels Beginner &
   python run_scenario_tests.py --runs 1 --levels Intermediate &
   ```

2. **Caching**:
   - BERT embeddings are cached automatically
   - Consider using Redis for distributed caching

3. **Resource Management**:
   - Monitor GPU memory usage for BERT
   - Implement rate limiting for OpenAI API calls

## Extending the System

### Adding New Chatbot Types

1. Create a new chatbot class implementing the required interface:
   ```python
   class YourChatBot:
       def __init__(self, config):
           # Initialize your chatbot
           pass
       
       def clear_history(self):
           # Clear conversation history
           pass
       
       def test(self, user_input, data_content, dict_content):
           # Generate response
           return response
   ```

2. Update the testing system to include your chatbot:
   ```python
   def run_all_scenarios(self, openai_chatbot, bert_chatbot, your_chatbot, num_runs=5):
       # Add your chatbot to the testing loop
   ```

### Custom Scoring Algorithms

Override the `evaluate_response()` method in `ScenarioTester`:

```python
def evaluate_response(self, expected: str, actual: str) -> int:
    # Implement your custom scoring logic
    # Return score from 1-10
    pass
```

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests for new functionality
4. Update documentation
5. Test with both OpenAI and BERT implementations

## License

This testing system is part of the AI-Chatbot project and follows the same licensing terms. 