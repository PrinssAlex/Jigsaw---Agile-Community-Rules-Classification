# Jigsaw - Agile (Reddit) Community Rules Classification

![Jigsaw Agile Community](img1.jpg)

##  Overview

Jigsaw - Agile Community is an advanced natural language processing ML project carried out to automatically detect rule violations in `Reddit` community discussions. The solution leverages state-of-the-art transformer models combined with strategic data processing techniques to identify content that violates community guidelines with high accuracy.

This project was developed to address the challenges of content moderation at scale, where manual review becomes impractical as community size grows. The system can be integrated into community platforms to flag potential violations for human review or to automatically moderate content based on predefined rules.

##  Problem Statement

Online communities face significant challenges in maintaining healthy discussion environments due to:
- The volume of user-generated content that needs review
- The nuanced nature of rule violations (often context-dependent)
- The resource-intensive nature of manual moderation
- The need for consistent application of rules across diverse content

Traditional keyword-based moderation systems fail to capture the contextual understanding required to accurately identify violations, leading to high false positive and false negative rates.

## Solution Approach

The project implements a sophisticated NLI (Natural Language Inference) framework that treats rule violation detection as a semantic relationship problem between content and rule statements. The system:

1. **Repurposes pre-trained language models** for rule violation detection
2. **Applies strategic data augmentation** to overcome limited training data
3. **Uses test-time augmentation** with multiple hypothesis formulations
4. **Integrates domain knowledge** through rule-informed post-processing

The solution achieves significant accuracy while maintaining computational efficiency, making it suitable for real-world application.

## ✨ Key Features

- **NLI-based violation detection**: Uses semantic relationship analysis between content and rule statements
- **Adaptive hypothesis formulation**: Multiple phrasing approaches to improve prediction robustness
- **Rule-specific post-processing**: Domain knowledge integration for refined predictions
- **Strategic data processing**: Effective utilization of limited training data
- **CUDA-optimized inference**: Efficient GPU utilization for scalable deployment
- **Memory-efficient design**: Careful resource management for production environments

##  Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA GPU with at least 8GB VRAM (recommended)

### Installation Steps
```bash
# Clone the repository
git clone 'repo url'
cd project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for CUDA memory management
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> ~/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Training
```bash
python train.py \
  --data-path /path/to/dataset \
  --model-name bert-large-uncased \
  --output-dir ./models/project \
  --epochs 5 \
  --batch-size 2
```

### Inference
```python
from inference import predict_violation

# Predict violation probability for a comment against a specific rule
probability = predict_violation(
    comment="Check out my free stream here: http://bit.ly",
    rule="No Advertising: Spam, referral links, unsolicited promotional content"
)

print(f"Violation probability: {probability:.4f}")
# Output: Violation probability: 0.9523
```

### Batch Processing
```bash
python batch_predict.py \
  --input-csv comments.csv \
  --output-csv predictions.csv \
  --model-path ./models/project
```

## Results

The project achieved several results during numerous trainings (which is still on-going) with the highest as of the time of uploading this README being 90.5%

## Future Work

Key areas for future development include:

- **Rule-embedding module**: To better capture semantic relationships between rules
- **Confidence calibration system**: Dynamic threshold adjustment by rule category
- **Multilingual support**: Expansion to non-English content moderation
- **Explainability features**: Providing justification for violation predictions
- **Active learning integration**: Prioritizing content that would be most valuable to label

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Hugging Face Transformers team for their excellent library
- The PyTorch team for their powerful deep learning framework
- The creators of the Jigsaw dataset for providing valuable training data

