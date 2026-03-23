"""
Download embedding model for offline use.
Run this ONCE with firewall/antivirus temporarily disabled.
"""
import os

# Ensure HuggingFace hub can access the internet
os.environ['HF_HUB_OFFLINE'] = '0'

from sentence_transformers import SentenceTransformer

print("Downloading BAAI/bge-small-en-v1.5...")
print("(This may take a few minutes — ~130MB)")
print("Make sure your firewall/antivirus is NOT blocking HuggingFace downloads")
print()

try:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("\n✓ Model downloaded and cached successfully!")
    print(f"✓ Embedding dimension: {model.get_sentence_embedding_dimension()} dimensions")
    print("\nYou can now run the pipeline scripts:")
    print("  python src/vectorstore.py")
    print("  python src/pipeline.py")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Temporarily disable firewall/antivirus")
    print("3. Check if HuggingFace is accessible: open https://huggingface.co in your browser")
    print("4. If behind a proxy, configure pip to use it")

