#!/usr/bin/env python3
"""
SmartSpend Project Setup Script
Automates the complete project setup and initial run
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class SmartSpendSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.steps_completed = 0
        self.total_steps = 8
        
    def print_header(self):
        print("="*60)
        print("🚀 SmartSpend Financial Intelligence Setup")
        print("="*60)
        print("This script will set up your complete SmartSpend project")
        print("Estimated time: 3-5 minutes")
        print("="*60)
    
    def print_step(self, step_name):
        self.steps_completed += 1
        print(f"\n[{self.steps_completed}/{self.total_steps}] 🔄 {step_name}...")
    
    def print_success(self, message):
        print(f"✅ {message}")
    
    def print_error(self, message):
        print(f"❌ {message}")
    
    def print_warning(self, message):
        print(f"⚠️ {message}")
    
    def create_directory_structure(self):
        """Create the project directory structure"""
        self.print_step("Creating directory structure")
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/sample",
            "models",
            "notebooks",
            "src",
            "tests",
            "docs",
            "docs/images"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files for empty directories
            if directory.startswith("data/"):
                gitkeep_file = dir_path / ".gitkeep"
                gitkeep_file.touch()
        
        self.print_success("Directory structure created")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        self.print_step("Checking Python version")
        
        if sys.version_info < (3, 8):
            self.print_error("Python 3.8+ is required. Please upgrade your Python installation.")
            sys.exit(1)
        
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.print_success(f"Python {python_version} is compatible")
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        self.print_step("Setting up virtual environment")
        
        venv_path = self.project_root / "smartspend_env"
        
        if not venv_path.exists():
            try:
                subprocess.run([sys.executable, "-m", "venv", "smartspend_env"], 
                             check=True, capture_output=True)
                self.print_success("Virtual environment created")
            except subprocess.CalledProcessError as e:
                self.print_error(f"Failed to create virtual environment: {e}")
                sys.exit(1)
        else:
            self.print_success("Virtual environment already exists")
        
        # Provide activation instructions
        if os.name == 'nt':  # Windows
            activate_cmd = "smartspend_env\\Scripts\\activate"
        else:  # Mac/Linux
            activate_cmd = "source smartspend_env/bin/activate"
        
        print(f"💡 To activate the environment, run: {activate_cmd}")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_step("Installing Python dependencies")
        
        # Check if we're in a virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.print_warning("Not in a virtual environment. Installing globally.")
        
        try:
            # Create requirements.txt if it doesn't exist
            requirements_path = self.project_root / "requirements.txt"
            if not requirements_path.exists():
                self.print_error("requirements.txt not found. Please create it first.")
                return
            
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            self.print_success("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            self.print_warning("Try running: pip install -r requirements.txt manually")
    
    def generate_sample_data(self):
        """Generate sample financial data"""
        self.print_step("Generating sample financial data")
        
        try:
            result = subprocess.run([sys.executable, "generate_sample_data.py"], 
                                  check=True, capture_output=True, text=True)
            self.print_success("Sample data generated")
            print(f"📊 {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to generate sample data: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
    
    def process_data(self):
        """Process and clean the data"""
        self.print_step("Processing and cleaning data")
        
        try:
            result = subprocess.run([sys.executable, "data_processor.py"], 
                                  check=True, capture_output=True, text=True)
            self.print_success("Data processed successfully")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to process data: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
    
    def train_models(self):
        """Train machine learning models"""
        self.print_step("Training machine learning models")
        
        try:
            result = subprocess.run([sys.executable, "ml_models.py"], 
                                  check=True, capture_output=True, text=True)
            self.print_success("ML models trained successfully")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to train models: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
    
    def create_demo_notebook(self):
        """Create a demo Jupyter notebook"""
        self.print_step("Creating demo notebook")
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# SmartSpend Financial Analysis Demo\n",
                        "\n",
                        "This notebook demonstrates the key features of the SmartSpend system.\n",
                        "\n",
                        "## Features Covered:\n",
                        "- Data loading and exploration\n",
                        "- Financial metrics calculation\n",
                        "- ML model predictions\n",
                        "- Visualization examples"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Import libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from datetime import datetime\n",
                        "\n",
                        "# Load the processed data\n",
                        "df = pd.read_csv('../data/processed/processed_transactions.csv')\n",
                        "df['DateTime'] = pd.to_datetime(df['DateTime'])\n",
                        "\n",
                        "print(f\"📊 Loaded {len(df)} transactions\")\n",
                        "print(f\"📅 Date range: {df['DateTime'].min()} to {df['DateTime'].max()}\")\n",
                        "df.head()"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Quick financial summary\n",
                        "total_income = df[df['Amount'] > 0]['Amount'].sum()\n",
                        "total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())\n",
                        "net_flow = total_income - total_expenses\n",
                        "\n",
                        "print(f\"💵 Total Income: ${total_income:,.2f}\")\n",
                        "print(f\"💸 Total Expenses: ${total_expenses:,.2f}\")\n",
                        "print(f\"💰 Net Flow: ${net_flow:,.2f}\")\n",
                        "print(f\"📈 Savings Rate: {((net_flow/total_income)*100):.1f}%\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        notebook_path = self.project_root / "notebooks" / "demo_analysis.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        self.print_success("Demo notebook created")
    
    def run_dashboard(self):
        """Launch the Streamlit dashboard"""
        self.print_step("Preparing to launch dashboard")
        
        dashboard_path = self.project_root / "streamlit_app.py"
        if not dashboard_path.exists():
            self.print_error("streamlit_app.py not found!")
            return
        
        self.print_success("Setup completed! 🎉")
        print("\n" + "="*60)
        print("🌟 SmartSpend is ready to use!")
        print("="*60)
        print("\n📊 Your project includes:")
        print("  ✅ Sample financial data (1,500+ transactions)")
        print("  ✅ Trained ML models (predictions & anomaly detection)")
        print("  ✅ Interactive dashboard")
        print("  ✅ Jupyter notebook for analysis")
        print("  ✅ Complete documentation")
        
        print("\n🚀 Next steps:")
        print("  1. Activate virtual environment:")
        if os.name == 'nt':
            print("     smartspend_env\\Scripts\\activate")
        else:
            print("     source smartspend_env/bin/activate")
        
        print("  2. Launch the dashboard:")
        print("     streamlit run streamlit_app.py")
        
        print("  3. Open browser to: http://localhost:8501")
        
        print("\n💡 Pro tips:")
        print("  • Upload your own CSV data for real insights")
        print("  • Check out the demo notebook in notebooks/")
        print("  • Customize the ML models in ml_models.py")
        print("  • Deploy to Streamlit Cloud for sharing")
        
        # Ask if user wants to launch now
        print("\n" + "="*60)
        launch = input("🤔 Would you like to launch the dashboard now? (y/n): ").lower().strip()
        
        if launch in ['y', 'yes']:
            print("\n🚀 Launching SmartSpend dashboard...")
            time.sleep(1)
            
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                             check=False)
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped. Thanks for using SmartSpend!")
            except Exception as e:
                print(f"\n❌ Error launching dashboard: {e}")
                print("💡 Try running manually: streamlit run streamlit_app.py")
    
    def setup_git(self):
        """Initialize git repository"""
        print("\n📝 Git Setup (optional)")
        setup_git = input("Initialize Git repository? (y/n): ").lower().strip()
        
        if setup_git in ['y', 'yes']:
            try:
                # Check if git is available
                subprocess.run(["git", "--version"], check=True, capture_output=True)
                
                # Initialize repo if not already initialized
                if not (self.project_root / ".git").exists():
                    subprocess.run(["git", "init"], check=True, capture_output=True)
                    print("✅ Git repository initialized")
                    
                    # Add all files
                    subprocess.run(["git", "add", "."], check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Initial SmartSpend project setup"], 
                                 check=True, capture_output=True)
                    print("✅ Initial commit created")
                    
                    print("💡 Next steps for GitHub:")
                    print("  1. Create a new repository on GitHub")
                    print("  2. git remote add origin <your-repo-url>")
                    print("  3. git push -u origin main")
                else:
                    print("✅ Git repository already exists")
                    
            except subprocess.CalledProcessError:
                print("⚠️ Git not found. Install Git to enable version control.")
            except Exception as e:
                print(f"⚠️ Git setup failed: {e}")
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        try:
            self.check_python_version()
            self.create_directory_structure()
            self.create_virtual_environment()
            self.install_dependencies()
            self.generate_sample_data()
            self.process_data()
            self.train_models()
            self.create_demo_notebook()
            
            self.setup_git()
            self.run_dashboard()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Setup interrupted by user")
            print("You can resume setup by running this script again")
        except Exception as e:
            print(f"\n❌ Setup failed with error: {e}")
            print("Please check the error details and try again")

def main():
    """Main setup function"""
    setup = SmartSpendSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()