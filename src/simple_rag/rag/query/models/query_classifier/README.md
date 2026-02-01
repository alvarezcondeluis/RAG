---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: How to learn Spanish?
- text: Latest football scores
- text: What is the net income ratio for VTI?
- text: What trust issues VTI?
- text: Get the costs per 10k for BND
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label         | Examples                                                                                                                                                                                              |
|:--------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| not related   | <ul><li>"What's the weather like today?"</li><li>'How do I cook pasta?'</li><li>'Tell me a joke'</li></ul>                                                                                            |
| simple query  | <ul><li>'What is the ticker for VTI?'</li><li>'Show me the expense ratio of VOO'</li><li>'What is the name of fund with ticker BND?'</li></ul>                                                        |
| complex query | <ul><li>'Show me all funds with expense ratio less than 0.1%'</li><li>'List funds managed by Vanguard with more than 1000 holdings'</li><li>'What are the top 5 holdings in VTI by weight?'</li></ul> |
| vector search | <ul><li>'Find funds with conservative investment strategy'</li><li>'What funds focus on growth investing?'</li><li>'Show me funds with low risk profile'</li></ul>                                    |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("How to learn Spanish?")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 6.8553 | 13  |

| Label         | Training Sample Count |
|:--------------|:----------------------|
| not related   | 40                    |
| simple query  | 39                    |
| complex query | 40                    |
| vector search | 40                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0008 | 1    | 0.3456        | -               |
| 0.0422 | 50   | 0.207         | -               |
| 0.0844 | 100  | 0.0937        | -               |
| 0.1266 | 150  | 0.0092        | -               |
| 0.1688 | 200  | 0.0019        | -               |
| 0.2110 | 250  | 0.0008        | -               |
| 0.2532 | 300  | 0.0007        | -               |
| 0.2954 | 350  | 0.0005        | -               |
| 0.3376 | 400  | 0.0004        | -               |
| 0.3797 | 450  | 0.0004        | -               |
| 0.4219 | 500  | 0.0003        | -               |
| 0.4641 | 550  | 0.0003        | -               |
| 0.5063 | 600  | 0.0003        | -               |
| 0.5485 | 650  | 0.0003        | -               |
| 0.5907 | 700  | 0.0003        | -               |
| 0.6329 | 750  | 0.0003        | -               |
| 0.6751 | 800  | 0.0008        | -               |
| 0.7173 | 850  | 0.0003        | -               |
| 0.7595 | 900  | 0.0003        | -               |
| 0.8017 | 950  | 0.0002        | -               |
| 0.8439 | 1000 | 0.0002        | -               |
| 0.8861 | 1050 | 0.0002        | -               |
| 0.9283 | 1100 | 0.0002        | -               |
| 0.9705 | 1150 | 0.0002        | -               |

### Framework Versions
- Python: 3.10.18
- SetFit: 1.1.3
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.8.0+cu128
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->