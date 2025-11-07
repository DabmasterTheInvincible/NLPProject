# AI Text Detection System

A robust sentence-level AI text detection system that classifies text as human-written, AI-generated, or paraphrased AI using a multi-module ensemble approach.

## Overview

This system combines four complementary modules:
- **DeBERTa-based Neural Classifier**: Contextual understanding and semantic analysis
- **Stylometric Analyzer**: Writing style profiling (lexical diversity, POS patterns, burstiness)
- **Perplexity-Entropy Module**: Text predictability measurement using language models
- **Semantic Analyzer**: E5 embeddings with FAISS similarity search

The system achieved 99% accuracy on the RAID dataset, outperforming state-of-the-art models like RADAR-Vicuna-7B and Hello-SimpleAI/chatgpt-detector-roberta.

## Features

- Real-time sentence-level classification
- Three-class detection: Human, AI-generated, and Paraphrased AI
- Multi-module ensemble approach for robust detection
- Web-based interface using Streamlit
- Explainable features including stylometry, perplexity, and semantic similarity

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DabmasterTheInvincible/NLPProject.git
cd NLPProject
```

### 2. Create a Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the following packages manually:

```bash
pip install streamlit torch transformers nltk spacy textstat faiss-cpu sentence-transformers pandas numpy scikit-learn
```

### 4. Download Required Models and Data

#### Download spaCy English Model:
```bash
python -m spacy download en_core_web_sm
```

#### Download NLTK Data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
```

### 5. Prepare Model Files

Ensure the following files are in the correct directories:
- Pre-trained DeBERTa model checkpoint in `models/` directory
- FAISS index files in `indices/` directory
- Any other required model files as specified in the codebase

## Running the Application

Once all dependencies are installed and models are in place, run:

```bash
streamlit run app.py
```

The application will start and open in your default web browser at `http://localhost:8501`

## Usage

1. Navigate to the web interface
2. Enter or paste the text you want to analyze or upload file to analyze
3. Click the "Analyze" button
4. View the classification results

## Project Structure

```
NLPProject/
│
├── indices/              # FAISS index files
├── models/              # Pre-trained model checkpoints
├── app.py               # Main Streamlit application
├── deberta.py           # DeBERTa classifier module
├── stylometry.py        # Stylometric feature extraction
├── perplexity.py        # Perplexity and entropy analysis
├── semantic.py          # Semantic similarity analysis
├── processing.py        # Text preprocessing utilities
├── _replace_ui.py       # UI customization
├── requirements.txt     # Python dependencies
└── README.md           # This file
```
---

**Note**: Ensure you have proper permissions and model files before running the application. Some models may require separate download due to size constraints.
