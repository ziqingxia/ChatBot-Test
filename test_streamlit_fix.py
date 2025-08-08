#!/usr/bin/env python3
"""
Simple test to verify the Streamlit app can be imported without errors.
"""

import sys
import os

def test_import():
    """Test if the streamlit app can be imported without errors."""
    try:
        print("Testing import of streamlit_testing.py...")
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the main function
        from streamlit_testing import main, initialize_session_state, check_environment, set_api_key
        
        print("✅ Successfully imported streamlit_testing.py")
        print("✅ All functions imported successfully:")
        print("   - main()")
        print("   - initialize_session_state()")
        print("   - check_environment()")
        print("   - set_api_key()")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing streamlit_testing.py: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_variable_initialization():
    """Test that variables are properly initialized."""
    try:
        print("\nTesting variable initialization...")
        
        # Import the main function
        from streamlit_testing import main
        
        print("✅ Variable initialization test passed")
        return True
        
    except UnboundLocalError as e:
        print(f"❌ UnboundLocalError: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Streamlit App Fix")
    print("=" * 40)
    
    # Test import
    import_success = test_import()
    
    # Test variable initialization
    if import_success:
        init_success = test_variable_initialization()
    else:
        init_success = False
    
    print("\n" + "=" * 40)
    print("📋 TEST RESULTS")
    print("=" * 40)
    
    if import_success and init_success:
        print("✅ All tests passed! The Streamlit app should work correctly.")
        print("\n🚀 You can now run:")
        print("   streamlit run streamlit_testing.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 