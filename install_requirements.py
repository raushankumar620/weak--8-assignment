import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    packages = [
        "streamlit>=1.28.0",
        "langchain>=0.0.350", 
        "langchain-community>=0.0.10",
        "chromadb>=0.4.15",
        "sentence-transformers>=2.2.2",
        "PyPDF2>=3.0.1",
        "python-docx>=0.8.11",
        "faiss-cpu>=1.7.4",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.1"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

if __name__ == "__main__":
    install_requirements()
    print("Installation complete!")
