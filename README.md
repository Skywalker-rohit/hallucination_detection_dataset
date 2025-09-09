# Hallucination Detection Dataset / Hallucinated Entity Recognition (HER) Dataset

**A comprehensive, cross-domain dataset with span-level annotations for training fine-grained hallucination detection models**

## 🎯 Overview

As Large Language Models (LLMs) become increasingly prevalent, hallucinations in their outputs pose serious threats to society by propagating misinformation. This repository contains the first cross-domain, balanced dataset specifically designed for training span-level hallucination detection models.

### Key Features

- **22,912 labeled samples** across **6,028 topics**
- **8 diverse domains**: Politics, Science, Healthcare, Law, History, Geography, Sports, Media  
- **Span-level annotations** with supporting evidence
- **Challenging hallucinations** (only 17% human detection accuracy)
- **Balanced dataset** with both true and false claims
- **Domain-agnostic generation pipeline**

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Claims | 22,912 |
| True Claims | 13,031 (56.9%) |
| False Claims | 9,881 (43.1%) |
| Topics Covered | 6,028 |
| Domains | 8 |
| Human Detection Accuracy | 17% |

## 🏗️ Dataset Structure

### Fields Description

```python
{
    "context": str,      # Text that needs hallucination checking
    "evidence": str,     # Supporting/refuting evidence
    "hf": bool,         # Hallucination flag (1=hallucinated, 0=true)
    "hinfo": {          # Detailed hallucination information
        "hp": str,       # Hallucinated phrases
        "hs": [int, int], # Character spans of hallucinations
        "pp": [str, str], # Original→replacement phrase mappings
        "cp": [str, str], # Context→paraphrase sentence mappings  
        "sf": [str]      # Sample factual phrases
    }
}
```

### Example Entry

```json
{
    "context": "Following these withdrawals the Bid Central Committee met on 9 June 2017 in Lausanne Switzerland to discuss the 2024 and 2028 bid processes",
    "evidence": "Following these withdrawals the IOC Executive Board met on 9 June 2017 in Lausanne Switzerland to discuss the 2024 and 2028 bid processes", 
    "hf": 1,
    "hinfo": {
        "hp": "Bid Central Committee",
        "hs": [28, 49],
        "pp": ["IOC Executive Board", "Bid Central Committee"],
        "cp": ["original sentence", "paraphrased evidence"],
        "sf": ["2024 and 2028 bid processes"]
    }
}
```

## 🔬 Methodology

Our synthetic data generation pipeline addresses key limitations in existing datasets:

### Pipeline Components

1. **Claim Extraction** - Identifies verifiable claims using YAKE-based classification
2. **Key-phrase Extraction** - Extracts objective, replaceable phrases  
3. **Subjective Filtering** - Removes opinion-based content using POS tagging + SentiWordNet
4. **Semantic Replacement** - Generates plausible alternatives via masked language modeling
5. **Entailment Verification** - Ensures semantic contradiction using RoBERTa-MNLI
6. **Evidence Generation** - Creates supporting evidence through controlled paraphrasing

### Quality Assurance

- **Textual Entailment Check**: Ensures replacements create genuine contradictions
- **Semantic Reranking**: Maximizes semantic distance while preserving plausibility  
- **Human Evaluation**: 17% detection accuracy confirms challenging, realistic hallucinations

## 📈 Evaluation Results

### Human Study Results
- **Overall Accuracy**: 17% (challenging for humans)
- **Hallucination Detection**: 26% when hallucination present
- **True Negative Rate**: 8% when no hallucination present

### Evidence Quality Distribution
| Quality Level | Count | Percentage | Jaccard Score Range |
|---------------|-------|------------|-------------------|
| High | 451 | 34.1% | ≥ 0.83 |
| Medium | 782 | 57.3% | 0.55 - 0.83 |
| Low | 131 | 9.6% | < 0.55 |

## 🚀 Getting Started

### Installation

```bash
pip install datasets transformers torch
```

### Loading the Dataset

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("your-username/hallucination-detection-dataset")

# Access training data
train_data = dataset['train']
test_data = dataset['test']

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
```

### Basic Usage Example

```python
import json

# Load a sample
sample = train_data[0]

print("Context:", sample['context'])
print("Evidence:", sample['evidence']) 
print("Contains Hallucination:", bool(sample['hf']))

if sample['hf']:
    hinfo = json.loads(sample['hinfo'])
    print("Hallucinated Phrase:", hinfo['hp'])
    print("Character Span:", hinfo['hs'])
```

## 🧪 Baseline Models

We provide baseline implementations for:

- **Span-level Detection**: Identifies hallucinated text segments
- **Binary Classification**: Determines presence of hallucinations
- **Evidence-based Verification**: Leverages supporting evidence

### Training Example

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model for span-level detection
model_name = "roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

# BIO tagging: B-HAL, I-HAL, O
# Your training code here...
```

## 📚 Domain Coverage

| Domain | Topics | Description |
|--------|--------|-------------|
| **Politics** | ~750 | Government, elections, policies |
| **Science** | ~750 | Research papers, discoveries, theories |
| **Healthcare** | ~750 | Medical conditions, treatments, research |
| **Law** | ~750 | Legal cases, regulations, procedures |
| **History** | ~750 | Historical events, figures, dates |
| **Geography** | ~750 | Countries, landmarks, demographics |
| **Sports** | ~750 | Athletes, competitions, records |
| **Media** | ~750 | News, entertainment, broadcasting |

## 🔄 Mathematical Formalism

The paper introduces a formal framework for efficient hallucination detection:

### Problem Formulation
- **Claims Decomposition**: Text T → Claims C = {c₁, c₂, ..., cₙ}
- **Atomic Claims**: Each claim cᵢ → Atomic claims {Aᵢⱼ}
- **Evidence Retrieval**: Optimize weighted coverage with budget constraints
- **Iterative Verification**: Update verification status until convergence

### Cost-Sensitive Detection
```
E* = arg max Σ w(Aᵢⱼ)
     E'⊆E    Aᵢⱼ∈⋃ₑₖ∈E' C(eₖ)
     
subject to |E'| ≤ k
```

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@article{kumar2025synthetic,
  title={Generating Synthetic Data for Hallucination Detection in LLMs},
  author={Kumar, Rohit and Chatterjee, Niladri},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Ways to Contribute
- Report bugs or dataset issues
- Suggest improvements to the generation pipeline
- Add new domain coverage
- Improve baseline models
- Submit evaluation results

## 📄 License

This dataset is released under the [MIT License](LICENSE). Please ensure compliance with source data licensing when using.

## 🙏 Acknowledgments

- Data sources: Wikipedia, Britannica, Stanford Encyclopedia of Philosophy, Microsoft WikiQA, ArnetMiner, CNN, Daily Mail
- Built using: SpaCy, BERT, RoBERTa, Sentence-BERT, YAKE
- Human evaluation conducted via Amazon Mechanical Turk

## 📞 Contact

- **Rohit Kumar**: srz248004@iitd.ac.in
- **Niladri Chatterjee**: niladri@maths.iitd.ac.in
- **Institution**: Indian Institute of Technology Delhi

## 🔗 Related Work

- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [FEVER Dataset](https://fever.ai/)
- [SciFact](https://scifact.apps.allenai.org/)
- [HaDeS](https://github.com/microsoft/HaDeS)

---

⭐ **Star this repository if you find it useful for your research!**
