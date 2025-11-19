# Embeddings at Scale - Code Examples

This repository contains all Python code examples from the book *Embeddings at Scale*. The examples are organized by chapter, with **253 code files** totaling **66,908 lines of code**.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- (Optional) GPU with CUDA for deep learning examples

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support (instead of faiss-cpu):
pip install faiss-gpu>=1.7.4
```

## Repository Structure

```
code_examples/
├── requirements.txt          # All Python dependencies
├── README.md                 # This file
├── ch01_foundations/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 16 code files
├── ch02_strategic_architecture/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 30 code files
├── ch03_vector_database_fundamentals/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 21 code files
├── ch04_custom_embedding_strategies/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 19 code files
├── ch05_contrastive_learning/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 21 code files
├── ch06_siamese_networks/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 13 code files
├── ch07_self_supervised_learning/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 8 code files
├── ch08_advanced_embedding_techniques/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 11 code files
├── ch09_embedding_pipeline_engineering/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 6 code files
├── ch10_scaling_embedding_training/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 10 code files
├── ch11_high_performance_vector_ops/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch12_data_engineering/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch13_rag_at_scale/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch14_semantic_search/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch15_recommendation_systems/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch16_anomaly_detection_security/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch17_automated_decision_systems/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch18_financial_services/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch19_healthcare_life_sciences/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch20_retail_ecommerce/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch21_manufacturing_industry40/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch22_media_entertainment/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch23_performance_optimization/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch24_security_privacy/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch25_monitoring_observability/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch26_future_trends/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 9 code files
├── ch27_organizational_transformation/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
├── ch28_implementation_roadmap/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 4 code files
├── ch30_path_forward/
│   ├── README.md            # Chapter-specific documentation
│   └── *.py                 # 5 code files
```

## Chapters Overview

### Part I: Foundations (Chapters 1-4)

**Chapter 1**: The Embedding Revolution - Fundamental examples demonstrating embeddings vs traditional approaches
- 16 code files in `ch01_foundations/`

**Chapter 2**: Strategic Architecture - Multi-modal embeddings, governance, cost optimization patterns
- 30 code files in `ch02_strategic_architecture/`

**Chapter 3**: Vector Database Fundamentals - HNSW, IVF-PQ, sharding, and trillion-scale indexing
- 21 code files in `ch03_vector_database_fundamentals/`

**Chapter 4**: Custom Embedding Strategies - Fine-tuning, multi-objective design, dimensionality optimization
- 19 code files in `ch04_custom_embedding_strategies/`

### Part II: Training Techniques (Chapters 5-10)

**Chapter 5**: Contrastive Learning - InfoNCE, SimCLR, MoCo, hard negative mining
- 21 code files in `ch05_contrastive_learning/`

**Chapter 6**: Siamese Networks - One-shot learning, triplet loss, threshold calibration
- 13 code files in `ch06_siamese_networks/`

**Chapter 7**: Self-Supervised Learning - Masked language models, autoencoders, domain-specific pre-training
- 8 code files in `ch07_self_supervised_learning/`

**Chapter 8**: Advanced Techniques - Hyperbolic embeddings, dynamic embeddings, federated learning
- 11 code files in `ch08_advanced_embedding_techniques/`

**Chapter 9**: Pipeline Engineering - Hybrid systems, deployment strategies, production patterns
- 6 code files in `ch09_embedding_pipeline_engineering/`

**Chapter 10**: Scaling Training - Distributed training, gradient accumulation, memory optimization
- 10 code files in `ch10_scaling_embedding_training/`

### Part III: Infrastructure & Operations (Chapters 11-12)

**Chapter 11**: High Performance Vector Operations - GPU acceleration, memory-mapped stores
- 5 code files in `ch11_high_performance_vector_ops/`

**Chapter 12**: Data Engineering - Data pipelines, quality monitoring, versioning
- 5 code files in `ch12_data_engineering/`

### Part IV: Applications (Chapters 13-22)

**Chapter 13**: RAG at Scale - Passage extraction, multi-stage retrieval, production RAG
- 5 code files in `ch13_rag_at_scale/`

**Chapter 14**: Semantic Search - Cross-modal search, query understanding, ranking
- 5 code files in `ch14_semantic_search/`

**Chapter 15**: Recommendation Systems - Embedding-based recommendations, cross-domain transfer
- 5 code files in `ch15_recommendation_systems/`

**Chapter 16**: Anomaly Detection & Security - Behavioral anomalies, security applications
- 5 code files in `ch16_anomaly_detection_security/`

**Chapter 17**: Automated Decision Systems - Embedding-driven automation
- 5 code files in `ch17_automated_decision_systems/`

**Chapter 18**: Financial Services - Finance-specific embedding applications
- 5 code files in `ch18_financial_services/`

**Chapter 19**: Healthcare & Life Sciences - Medical and healthcare embedding use cases
- 5 code files in `ch19_healthcare_life_sciences/`

**Chapter 20**: Retail & E-commerce - Product embeddings, style matching, demand forecasting
- 5 code files in `ch20_retail_ecommerce/`

**Chapter 21**: Manufacturing & Industry 4.0 - Industrial IoT, predictive maintenance
- 5 code files in `ch21_manufacturing_industry40/`

**Chapter 22**: Media & Entertainment - Content embeddings, recommendation
- 5 code files in `ch22_media_entertainment/`

### Part V: Production & Future (Chapters 23-30)

**Chapter 23**: Performance Optimization - Caching, quantization, hardware acceleration
- 5 code files in `ch23_performance_optimization/`

**Chapter 24**: Security & Privacy - Differential privacy, secure aggregation, access control
- 5 code files in `ch24_security_privacy/`

**Chapter 25**: Monitoring & Observability - Metrics, drift detection, quality monitoring
- 5 code files in `ch25_monitoring_observability/`

**Chapter 26**: Future Trends - Quantum embeddings, neuromorphic computing, blockchain
- 9 code files in `ch26_future_trends/`

**Chapter 27**: Organizational Transformation - Capability development, governance frameworks
- 5 code files in `ch27_organizational_transformation/`

**Chapter 28**: Implementation Roadmap - Deployment stages, technology selection
- 4 code files in `ch28_implementation_roadmap/`

**Chapter 30**: Path Forward - Innovation stages, partnership models, transformation frameworks
- 5 code files in `ch30_path_forward/`

## How to Use These Examples

### 1. Complete, Runnable Examples

Most examples in Chapters 4-10 are **complete and runnable** (with proper setup):

```bash
cd ch05_contrastive_learning
python infonceloss.py  # Requires synthetic data or modifications
```

### 2. Illustrative Code Snippets

Some examples are **illustrative patterns** showing best practices:

- Architecture patterns (Chapter 2, 3)
- Production deployment patterns (Chapter 6, 9, 23)
- Domain-specific applications (Chapters 13-22)

### 3. Framework Examples

Framework code provides **reusable components** for building systems:

- `ch02_strategic_architecture/embeddingdatagovernance.py`
- `ch04_custom_embedding_strategies/embeddingtco.py`
- `ch06_siamese_networks/thresholdcalibrator.py`

## Important Notes

### Data Requirements

Most code examples assume you have:

- Training data (text, images, or structured data)
- Pre-trained models (from HuggingFace or similar)
- Vector databases (FAISS, Qdrant, Weaviate, etc.)

You will need to:

1. **Replace placeholder data** with your actual datasets
2. **Download pre-trained models** (many use HuggingFace models)
3. **Configure paths** and parameters for your environment

### GPU vs CPU

- **Deep learning examples** (Ch 5-8, 10) benefit from GPU
- **Vector operations** (Ch 3, 11) are much faster on GPU
- **Small examples** can run on CPU

For CPU-only:
```bash
# Most PyTorch code will fallback to CPU automatically
# Use faiss-cpu instead of faiss-gpu
```

## Key Technologies

**Deep Learning Frameworks**:
- torch>=2.0.0
- transformers>=4.30.0
- sentence-transformers>=2.2.0
- torchvision>=0.15.0

**Vector Databases & Search**:
- faiss-cpu>=1.7.4
- numpy>=1.24.0
- scikit-learn>=1.3.0

**Data Processing**:
- pandas>=2.0.0
- datasets>=2.14.0
- tokenizers>=0.13.0

**Visualization & Analysis**:
- matplotlib>=3.7.0
- scipy>=1.10.0

**NLP**:
- nltk>=3.8.0
- gensim>=4.3.0
- transformers>=4.30.0

**APIs & Services**:
- openai>=1.0.0

## Code Completeness Assessment

| Category | Chapters | Completeness | Notes |
|----------|----------|--------------|-------|
| Foundations | 1-3 | Illustrative | Demonstrates concepts, needs data |
| Custom Embeddings | 4 | High | Most examples runnable with setup |
| Training | 5-8 | High | Complete training loops, needs data |
| Siamese Networks | 6 | Very High | Production-ready patterns |
| Scaling | 9-10 | Medium | Requires distributed setup |
| Infrastructure | 11-12 | Medium | Patterns and frameworks |
| Applications | 13-22 | Illustrative | Domain-specific examples |
| Production | 23-25 | Medium | Monitoring and optimization patterns |
| Future | 26-30 | Conceptual | Emerging technologies |

## License

These code examples are from the book *Embeddings at Scale*. Please refer to the book for full context and explanations.

## Getting Help

For questions about the code:

1. **Read the corresponding book chapter** for full context
2. **Check the chapter README** for specific setup instructions
3. **Review the code comments** for implementation details

## Quick Start Examples

### Example 1: Fine-tune Sentence Embeddings

```python
# From ch04_custom_embedding_strategies/embeddingfinetuner.py
from sentence_transformers import SentenceTransformer

# Load base model
model = SentenceTransformer('all-mpnet-base-v2')

# Fine-tune on your domain data
# (See embeddingfinetuner.py for complete example)
```

### Example 2: Train Siamese Network

```python
# From ch06_siamese_networks/siamesenetwork.py
import torch
import torch.nn as nn

# Create Siamese network
# (See siamesenetwork.py for complete architecture)
```

### Example 3: Contrastive Learning

```python
# From ch05_contrastive_learning/infonceloss.py
# Implement InfoNCE loss for self-supervised learning
# (See infonceloss.py for complete implementation)
```

