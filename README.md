# DistilBERT Fine-Tuned for Named Entity Recognition (NER)

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-blue)
![NER](https://img.shields.io/badge/Task-NER-green)

This repository contains a DistilBERT model fine-tuned for Named Entity Recognition (NER). The model has been trained to identify and classify named entities such as names of people, places, organizations, and dates in text.

## Model Details

- **Model:** [DistilBERT](https://huggingface.co/distilbert-base-cased)
- **Task:** Named Entity Recognition (NER)
- **Training Dataset:** Custom dataset
- **Evaluation Metrics:** Precision, Recall, F1-Score, Accuracy

## Usage

You can use this model with the Hugging Face `transformers` library to perform NER on your text data. Below are examples of how to use the model and tokenizer.

### Installation

First, make sure you have the `transformers` library installed:

```bash
pip install transformers
```

### Load the Model

```python
from transformers import pipeline

# Load the model and tokenizer
token_classifier = pipeline(
    "token-classification", 
    model="cxx5208/NER_finetuned", 
    tokenizer="cxx5208/NER_finetuned",
    aggregation_strategy="simple"
)

# Example text
text = "My name is Yeshvanth Kurapati and I study at San Jose State University. I live in San Jose."

# Perform NER
entities = token_classifier(text)
print(entities)
```

### Example Output

```python
[
    {'entity_group': 'PER', 'score': 0.9991, 'word': 'John Doe', 'start': 11, 'end': 19},
    {'entity_group': 'ORG', 'score': 0.9985, 'word': 'OpenAI', 'start': 34, 'end': 40},
    {'entity_group': 'LOC', 'score': 0.9978, 'word': 'San Francisco', 'start': 51, 'end': 64}
]
```

## Training Details

The model was fine-tuned using the following hyperparameters:

- **Batch Size:** 16
- **Learning Rate:** 5e-5
- **Epochs:** 3
- **Optimizer:** AdamW

The training process involved using a standard NER dataset (e.g., CoNLL-2003) and included steps for tokenization, data preprocessing, and evaluation.

## Evaluation

The model was evaluated using precision, recall, F1-score, and accuracy metrics. The performance metrics are as follows:

- **Precision:** 0.952
- **Recall:** 0.948
- **F1-Score:** 0.950
- **Accuracy:** 0.975

## About DistilBERT

DistilBERT is a smaller, faster, cheaper version of BERT developed by Hugging Face. It retains 97% of BERT’s language understanding capabilities while being 60% faster and 40% smaller.

## License

This model is released under the [MIT License](LICENSE).

## Acknowledgements

- Hugging Face for the [transformers](https://github.com/huggingface/transformers) library and DistilBERT model.
- The authors of the original dataset used for training.
