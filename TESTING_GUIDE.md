# Chatbot Scenario Testing Framework

This testing framework evaluates both OpenAI (GPT-4) and BERT-based chatbots across various radio communication scenarios to test flexibility and consistency.

## Overview

The framework tests the chatbots on:
- **Beginner Level**: Basic radio communication protocols
- **Intermediate Level**: TOA (Take Over Authority) procedures
- **Advanced Level**: Emergency response protocols

Each scenario is run 5 times to ensure consistency, and results are compared in a table format.

## Test Scenarios

### Beginner Level
- **Scenario 2.1**: Basic radio check with consistent naming (TPO Lim)
- **Scenario 2.2**: Radio check with different phrasing (TPO XX)
- **Scenario 2.3**: Control set with standard protocol

### Intermediate Level
- **Scenario 6.1**: TOA surrender with full protocol
- **Scenario 6.2**: TOA surrender with different phrasing
- **Scenario 6.3**: Control set with standard TOA protocol

### Advanced Level
- **Scenario 14.1**: Emergency rail crack response
- **Scenario 14.2**: Emergency response with different phrasing
- **Scenario 14.3**: Emergency response with full location details

## Installation

1. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. For BERT testing and Streamlit interface, install additional dependencies:
```bash
pip install langchain langchain-community sentence-transformers faiss-cpu plotly
```

3. Set up your OpenAI API key:
   - **Option A**: Set environment variable: `export OPENAI_API_KEY="your-api-key-here"`
   - **Option B**: Enter it directly in the Streamlit interface (recommended)

## Usage

### Option 1: Streamlit Web Interface (Recommended)
Launch the interactive web interface:
```bash
python run_streamlit_testing.py
```
Or manually:
```bash
streamlit run streamlit_testing.py
```

**Features:**
- 🖱️ **Interactive scenario selection** with checkboxes
- 📊 **Real-time progress tracking** with progress bars
- 📈 **Interactive charts and visualizations** using Plotly
- 📋 **Detailed results tables** with filtering options
- 💾 **Export functionality** (JSON, CSV)
- 🎨 **Beautiful UI** with custom styling

### Option 2: Command Line Interface
Run all scenarios with 5 iterations each:
```bash
python run_tests.py
```

### Custom Testing Options

**Run specific scenarios:**
```bash
python run_tests.py --scenarios "Scenario 2.1" "Scenario 6.1"
```

**Test specific difficulty levels:**
```bash
python run_tests.py --levels Beginner Intermediate
```

**Change number of test runs:**
```bash
python run_tests.py --runs 10
```

**Specify output directory:**
```bash
python run_tests.py --output-dir my_results
```

**Use custom config file:**
```bash
python run_tests.py --config my_config.yaml
```

### Command Line Options

- `--runs`: Number of test runs per scenario (default: 5)
- `--scenarios`: Specific scenarios to test
- `--levels`: Specific difficulty levels to test (Beginner/Intermediate/Advanced)
- `--output-dir`: Output directory for results (default: test_results)
- `--config`: Path to config file (default: configs/config.yaml)

## Output Format

### Console Output
The framework displays real-time progress and a final comparison table:

```
| Scenario | OpenAI | BERT |
|----------|--------|------|
| Scenario 2.1 | 3.2 | 4.1 | 3.8 | 3.5 | 3.9 | Total: 3.7 | 3.1 | 4.0 | 3.7 | 3.4 | 3.8 | Total: 3.6 |
| Scenario 2.2 | 3.0 | 3.8 | 3.5 | 3.2 | 3.7 | Total: 3.4 | 2.9 | 3.9 | 3.6 | 3.3 | 3.7 | Total: 3.5 |
...
```

### File Outputs

1. **Detailed Results** (`detailed_results_YYYYMMDD_HHMMSS.json`):
   - Complete conversation logs
   - Individual scores for each response
   - Model responses and expected responses
   - Metadata for each test run

2. **Comparison Table** (`comparison_table_YYYYMMDD_HHMMSS.txt`):
   - Formatted table showing scores for each run
   - Average scores per scenario
   - Easy-to-read comparison between OpenAI and BERT

## Scoring System

Each response is scored on a scale of 0-5 based on:

### Beginner Level Criteria
- Correct call sign usage (Kendrik)
- Proper radio protocol (send, over, out)
- Appropriate response to radio check
- Consistent naming throughout conversation
- Handling of different phrasing

### Intermediate Level Criteria
- Correct call sign usage (Main Line Zero Two)
- Proper TOA handling and acknowledgment
- Appropriate Line Clear protocol
- Time stamp handling (0415 hrs)
- Protocol adherence

### Advanced Level Criteria
- Correct call sign usage (Depot Zero)
- Proper emergency protocol handling
- Appropriate safety instructions
- Emergency response protocol adherence

## Understanding Results

### Score Interpretation
- **5.0**: Perfect response matching expected protocol
- **4.0-4.9**: Excellent response with minor variations
- **3.0-3.9**: Good response with some protocol adherence
- **2.0-2.9**: Fair response with basic understanding
- **1.0-1.9**: Poor response with limited protocol knowledge
- **0.0-0.9**: Incorrect or inappropriate response

### Consistency Analysis
- Compare scores across multiple runs
- Look for patterns in performance
- Identify scenarios where one model outperforms the other
- Check for consistency in naming and protocol adherence

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Enter your API key in the Streamlit interface sidebar
   - Ensure the key starts with "sk-" and is valid
   - Check API key permissions and quota
   - You can also set it as an environment variable: `export OPENAI_API_KEY="your-key"`

2. **BERT Dependencies Missing**:
   - Install required packages: `pip install langchain langchain-community sentence-transformers faiss-cpu`
   - BERT testing will be skipped if dependencies are unavailable

3. **Database Loading Errors**:
   - Ensure database files exist in the specified paths
   - Check file permissions and format

4. **Memory Issues**:
   - Reduce the number of concurrent tests
   - Use smaller embedding models for BERT

### Performance Tips

1. **Faster Testing**:
   - Use `--scenarios` to test specific scenarios only
   - Reduce `--runs` for quick testing
   - Test one model type at a time

2. **Better Results**:
   - Ensure your RAG database is well-populated
   - Use appropriate system prompts
   - Fine-tune scoring criteria if needed

## Customization

### Adding New Scenarios
Edit `test_scenarios.py` and add new scenarios to the `define_scenarios()` method:

```python
scenarios.append(TestScenario(
    name="Scenario X.Y",
    level="Beginner/Intermediate/Advanced",
    conversation=[
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Expected response"},
        # ... more conversation turns
    ],
    expected_responses=[
        "Expected response 1",
        "Expected response 2",
        # ... more expected responses
    ],
    scoring_criteria=[
        "Criterion 1",
        "Criterion 2",
        # ... more criteria
    ]
))
```

### Modifying Scoring
Adjust the `score_response()` method to change how responses are evaluated based on your specific needs.

### Custom Models
The framework can be extended to test other model types by implementing new test methods similar to `run_openai_test()` and `run_bert_test()`.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed error logs in the output files
3. Ensure all dependencies are properly installed
4. Verify your configuration file is correct 