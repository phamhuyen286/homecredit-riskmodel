"""
Setup script to install all dependencies for the Home Credit Risk Model project.
Run this script to ensure all required libraries are installed.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found.")
        return False
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("All required packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install required packages.")
        return False

def check_environment():
    """Check if all required packages are installed"""
    required_packages = [
        'polars', 'pandas', 'numpy', 'scikit-learn', 
        'lightgbm', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("All required packages are already installed!")
        return True

def main():
    """Main function to set up the environment"""
    print("Setting up environment for Home Credit Risk Model project...")
    
    # Check if required packages are already installed
    if not check_environment():
        # Install requirements if any package is missing
        install_requirements()
        
        # Verify installation
        if check_environment():
            print("Setup completed successfully!")
        else:
            print("Setup failed. Please install missing packages manually.")
    else:
        print("Setup completed successfully!")

if __name__ == "__main__":
    main()