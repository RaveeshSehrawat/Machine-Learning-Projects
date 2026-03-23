---
title: FinSight
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# FinSight — HuggingFace Spaces Configuration

This file contains the required configuration metadata for deploying FinSight on HuggingFace Spaces.

## Configuration Reference

| Field | Value | Description |
|-------|-------|-------------|
| **title** | FinSight | Display name of the Space |
| **emoji** | 📊 | Icon shown in Space browser |
| **colorFrom** | blue | Gradient start color |
| **colorTo** | green | Gradient end color |
| **sdk** | streamlit | Framework (Streamlit, Gradio, or Docker) |
| **sdk_version** | 1.28.0 | Streamlit version |
| **python_version** | 3.10 | Python version for the environment |
| **app_file** | app.py | Entry point for the application |
| **pinned** | false | Whether to pin the Space to your profile |

## Setup Instructions

1. **Create a new Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose a name (e.g., `finsight`)
   - Select **Streamlit** as the SDK
   - Set visibility to Public or Private

2. **Copy the YAML header** (lines 1-10) into a file named `README.md` at the root of your Space repository

3. **Push your code** to the Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/finsight
   cd finsight
   
   # Copy FinSight files
   cp -r /path/to/finsight/* .
   
   # Ensure this configuration is in README.md
   git add .
   git commit -m "Deploy FinSight on HuggingFace Spaces"
   git push
   ```

4. **Wait for build** (~2-3 minutes)
   - Space will auto-install dependencies from `requirements.txt`
   - Mistral 7B model downloads on first launch (~4GB)
   - Subsequent loads are instant

## Pre-built Artifacts

For fastest startup, pre-populate these directories in your Space repo:

```
.
├── chroma_db/              # Pre-built vector database
│   ├── chroma.sqlite3
│   └── 153bd3bf-9f90-44bf-997d-1a39934d701d/
├── data/
│   ├── embeddings.npy      # Pre-computed embeddings
│   └── SP500_Alpha_Dataset_Final.csv
└── requirements.txt        # Python dependencies
```

This eliminates the need for users to rebuild the vector database and embeddings on first access.

## Reference

Full HuggingFace Spaces documentation: https://huggingface.co/docs/hub/spaces-config-reference
