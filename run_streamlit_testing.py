#!/usr/bin/env python3
"""
Launcher script for the Streamlit testing interface.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit testing interface."""
    print("🚀 Launching Chatbot Testing Framework (Streamlit)")
    print("=" * 60)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
        print("✅ Streamlit installed")
    
    # Check if plotly is available
    try:
        import plotly
        print("✅ Plotly is available")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.0.0"])
        print("✅ Plotly installed")
    
    # Check environment
    print("ℹ️  Note: You can enter your OpenAI API key directly in the web interface")
    print("   No need to set it as an environment variable.")
    
    print("\n🌐 Starting Streamlit server...")
    print("📱 The interface will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use Ctrl+C to stop the server")
    print("   - Check the sidebar for environment status")
    print("   - Select scenarios and models before running tests")
    print("   - Results will be displayed in interactive charts and tables")
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_testing.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("💡 Try running manually: streamlit run streamlit_testing.py")

if __name__ == "__main__":
    main() 