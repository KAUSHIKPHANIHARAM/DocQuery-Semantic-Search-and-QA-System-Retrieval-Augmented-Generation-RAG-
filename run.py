#!/usr/bin/env python3
"""
Quick Start Script for PDF Question Answering System
Run this to launch the application with one command
"""

import subprocess
import sys
import os

def main():
    print("=" * 70)
    print("  PDF QUESTION-ANSWERING SYSTEM")
    print("  Quick Start Launcher")
    print("=" * 70)
    print()
    
    # Check Python version
    print("✓ Python version:", sys.version.split()[0])
    
    # Check if venv is activated
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found at:", venv_path)
        print("   Please create it first:")
        print("   python -m venv venv")
        sys.exit(1)
    
    print("✓ Virtual environment found")
    print()
    
    # Launch Streamlit
    print("🚀 Launching Streamlit application...")
    print("   App will open at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("-" * 70)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_production.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
