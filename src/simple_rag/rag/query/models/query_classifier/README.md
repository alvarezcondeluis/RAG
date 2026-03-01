---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Which funds from BlackRock hold companies with recent 10-K filings?
- text: How many holdings does VOO have?
- text: What is quantum physics?
- text: List all funds that invest more than 20% in Technology sector
- text: Which year had the worst total return for BND?
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
- **Number of Classes:** 9 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label               | Examples                                                                                                                                                                                                                             |
|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| not_related         | <ul><li>"What's the weather like today?"</li><li>'How do I cook pasta?'</li><li>'Tell me a joke'</li></ul>                                                                                                                           |
| fund_basic          | <ul><li>'What is the ticker for VTI?'</li><li>'Show me the expense ratio of VOO'</li><li>'What is the name of fund with ticker BND?'</li></ul>                                                                                       |
| fund_performance    | <ul><li>'What is the 1-year return for VTI?'</li><li>'Show me the 5-year return of VOO'</li><li>'What is the 10-year return for BND?'</li></ul>                                                                                      |
| fund_portfolio      | <ul><li>'What are the top 5 holdings in VTI by weight?'</li><li>'List funds with more than 1000 holdings'</li><li>'What sector has the highest allocation in VTI?'</li></ul>                                                         |
| fund_profile        | <ul><li>'Find funds with conservative investment strategy'</li><li>'What funds focus on growth investing?'</li><li>'Show me funds with low risk profile'</li></ul>                                                                   |
| company_filing      | <ul><li>'Show risk factors for Apple'</li><li>"What are the risk factors in Apple's 10-K?"</li><li>'Get the business description from AAPL 10-K filing'</li></ul>                                                                    |
| company_people      | <ul><li>'Who are the managers of VOO?'</li><li>'Show me all funds managed by person John Doe'</li><li>'Who is the CEO of Apple?'</li></ul>                                                                                           |
| hybrid_graph_vector | <ul><li>'Find low-risk funds managed by Vanguard'</li><li>'Show me conservative strategy funds with expense ratio below 0.1%'</li><li>'Find growth-focused funds with more than 500 holdings'</li></ul>                              |
| cross_entity        | <ul><li>'Which funds hold Apple stock and who manages them?'</li><li>'Show Vanguard funds that invest in companies with recent insider sales'</li><li>'What funds hold Microsoft and what does their 10-K say about risk?'</li></ul> |

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
preds = model("What is quantum physics?")
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
| Word count   | 3   | 7.5753 | 16  |

| Label               | Training Sample Count |
|:--------------------|:----------------------|
| not_related         | 50                    |
| fund_basic          | 48                    |
| fund_performance    | 31                    |
| fund_portfolio      | 31                    |
| fund_profile        | 52                    |
| company_filing      | 30                    |
| company_people      | 30                    |
| hybrid_graph_vector | 30                    |
| cross_entity        | 30                    |

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
| 0.0002 | 1    | 0.3371        | -               |
| 0.0082 | 50   | 0.2417        | -               |
| 0.0165 | 100  | 0.2227        | -               |
| 0.0247 | 150  | 0.203         | -               |
| 0.0329 | 200  | 0.1592        | -               |
| 0.0412 | 250  | 0.1219        | -               |
| 0.0494 | 300  | 0.0765        | -               |
| 0.0576 | 350  | 0.0708        | -               |
| 0.0658 | 400  | 0.0434        | -               |
| 0.0741 | 450  | 0.0235        | -               |
| 0.0823 | 500  | 0.0149        | -               |
| 0.0905 | 550  | 0.009         | -               |
| 0.0988 | 600  | 0.0035        | -               |
| 0.1070 | 650  | 0.0026        | -               |
| 0.1152 | 700  | 0.0019        | -               |
| 0.1235 | 750  | 0.0011        | -               |
| 0.1317 | 800  | 0.0009        | -               |
| 0.1399 | 850  | 0.001         | -               |
| 0.1481 | 900  | 0.0007        | -               |
| 0.1564 | 950  | 0.0006        | -               |
| 0.1646 | 1000 | 0.0006        | -               |
| 0.1728 | 1050 | 0.0005        | -               |
| 0.1811 | 1100 | 0.0004        | -               |
| 0.1893 | 1150 | 0.0004        | -               |
| 0.1975 | 1200 | 0.0004        | -               |
| 0.2058 | 1250 | 0.0004        | -               |
| 0.2140 | 1300 | 0.0004        | -               |
| 0.2222 | 1350 | 0.0004        | -               |
| 0.2305 | 1400 | 0.0003        | -               |
| 0.2387 | 1450 | 0.0003        | -               |
| 0.2469 | 1500 | 0.0003        | -               |
| 0.2551 | 1550 | 0.0003        | -               |
| 0.2634 | 1600 | 0.0003        | -               |
| 0.2716 | 1650 | 0.0003        | -               |
| 0.2798 | 1700 | 0.0003        | -               |
| 0.2881 | 1750 | 0.0003        | -               |
| 0.2963 | 1800 | 0.0003        | -               |
| 0.3045 | 1850 | 0.0002        | -               |
| 0.3128 | 1900 | 0.0002        | -               |
| 0.3210 | 1950 | 0.0003        | -               |
| 0.3292 | 2000 | 0.0004        | -               |
| 0.3374 | 2050 | 0.0029        | -               |
| 0.3457 | 2100 | 0.0023        | -               |
| 0.3539 | 2150 | 0.0004        | -               |
| 0.3621 | 2200 | 0.0003        | -               |
| 0.3704 | 2250 | 0.0003        | -               |
| 0.3786 | 2300 | 0.0002        | -               |
| 0.3868 | 2350 | 0.0002        | -               |
| 0.3951 | 2400 | 0.0002        | -               |
| 0.4033 | 2450 | 0.0002        | -               |
| 0.4115 | 2500 | 0.0006        | -               |
| 0.4198 | 2550 | 0.0002        | -               |
| 0.4280 | 2600 | 0.0002        | -               |
| 0.4362 | 2650 | 0.0002        | -               |
| 0.4444 | 2700 | 0.0002        | -               |
| 0.4527 | 2750 | 0.0002        | -               |
| 0.4609 | 2800 | 0.0002        | -               |
| 0.4691 | 2850 | 0.0002        | -               |
| 0.4774 | 2900 | 0.0002        | -               |
| 0.4856 | 2950 | 0.0016        | -               |
| 0.4938 | 3000 | 0.0002        | -               |
| 0.5021 | 3050 | 0.0002        | -               |
| 0.5103 | 3100 | 0.0002        | -               |
| 0.5185 | 3150 | 0.0002        | -               |
| 0.5267 | 3200 | 0.0002        | -               |
| 0.5350 | 3250 | 0.0001        | -               |
| 0.5432 | 3300 | 0.0001        | -               |
| 0.5514 | 3350 | 0.0001        | -               |
| 0.5597 | 3400 | 0.0001        | -               |
| 0.5679 | 3450 | 0.0001        | -               |
| 0.5761 | 3500 | 0.0001        | -               |
| 0.5844 | 3550 | 0.0001        | -               |
| 0.5926 | 3600 | 0.0001        | -               |
| 0.6008 | 3650 | 0.0001        | -               |
| 0.6091 | 3700 | 0.0001        | -               |
| 0.6173 | 3750 | 0.0001        | -               |
| 0.6255 | 3800 | 0.0001        | -               |
| 0.6337 | 3850 | 0.0001        | -               |
| 0.6420 | 3900 | 0.0001        | -               |
| 0.6502 | 3950 | 0.0001        | -               |
| 0.6584 | 4000 | 0.0001        | -               |
| 0.6667 | 4050 | 0.0001        | -               |
| 0.6749 | 4100 | 0.0001        | -               |
| 0.6831 | 4150 | 0.0001        | -               |
| 0.6914 | 4200 | 0.0001        | -               |
| 0.6996 | 4250 | 0.0001        | -               |
| 0.7078 | 4300 | 0.0001        | -               |
| 0.7160 | 4350 | 0.0001        | -               |
| 0.7243 | 4400 | 0.0001        | -               |
| 0.7325 | 4450 | 0.0001        | -               |
| 0.7407 | 4500 | 0.0001        | -               |
| 0.7490 | 4550 | 0.0001        | -               |
| 0.7572 | 4600 | 0.0001        | -               |
| 0.7654 | 4650 | 0.0001        | -               |
| 0.7737 | 4700 | 0.0001        | -               |
| 0.7819 | 4750 | 0.0001        | -               |
| 0.7901 | 4800 | 0.0001        | -               |
| 0.7984 | 4850 | 0.0001        | -               |
| 0.8066 | 4900 | 0.0001        | -               |
| 0.8148 | 4950 | 0.0001        | -               |
| 0.8230 | 5000 | 0.0001        | -               |
| 0.8313 | 5050 | 0.0001        | -               |
| 0.8395 | 5100 | 0.0001        | -               |
| 0.8477 | 5150 | 0.0001        | -               |
| 0.8560 | 5200 | 0.0001        | -               |
| 0.8642 | 5250 | 0.0001        | -               |
| 0.8724 | 5300 | 0.0001        | -               |
| 0.8807 | 5350 | 0.0001        | -               |
| 0.8889 | 5400 | 0.0001        | -               |
| 0.8971 | 5450 | 0.0001        | -               |
| 0.9053 | 5500 | 0.0001        | -               |
| 0.9136 | 5550 | 0.0001        | -               |
| 0.9218 | 5600 | 0.0001        | -               |
| 0.9300 | 5650 | 0.0001        | -               |
| 0.9383 | 5700 | 0.0001        | -               |
| 0.9465 | 5750 | 0.0001        | -               |
| 0.9547 | 5800 | 0.0001        | -               |
| 0.9630 | 5850 | 0.0001        | -               |
| 0.9712 | 5900 | 0.0001        | -               |
| 0.9794 | 5950 | 0.0001        | -               |
| 0.9877 | 6000 | 0.0001        | -               |
| 0.9959 | 6050 | 0.0001        | -               |

### Framework Versions
- Python: 3.11.14
- SetFit: 1.1.3
- Sentence Transformers: 5.2.2
- Transformers: 4.57.6
- PyTorch: 2.9.1+cu128
- Datasets: 4.3.0
- Tokenizers: 0.22.2

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