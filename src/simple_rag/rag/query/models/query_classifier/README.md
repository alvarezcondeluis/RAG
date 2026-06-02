---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: What is the NAV at year end for VSIAX?
- text: How many stocks are in the Vanguard 500 Index fund's portfolio?
- text: What is quantum physics?
- text: How does the Vanguard ESG International fund describe its investment approach?
- text: List funds with more than 2000 holdings
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: false
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A OneVsRestClassifier instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a OneVsRestClassifier instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 6 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

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
| Word count   | 3   | 8.9275 | 21  |

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
| 0.0001 | 1    | 0.2511        | -               |
| 0.0053 | 50   | 0.2313        | -               |
| 0.0106 | 100  | 0.207         | -               |
| 0.0159 | 150  | 0.2045        | -               |
| 0.0212 | 200  | 0.187         | -               |
| 0.0265 | 250  | 0.1688        | -               |
| 0.0318 | 300  | 0.1537        | -               |
| 0.0371 | 350  | 0.1281        | -               |
| 0.0425 | 400  | 0.1212        | -               |
| 0.0478 | 450  | 0.0876        | -               |
| 0.0531 | 500  | 0.0861        | -               |
| 0.0584 | 550  | 0.0705        | -               |
| 0.0637 | 600  | 0.0609        | -               |
| 0.0690 | 650  | 0.0586        | -               |
| 0.0743 | 700  | 0.0529        | -               |
| 0.0796 | 750  | 0.0447        | -               |
| 0.0849 | 800  | 0.0475        | -               |
| 0.0902 | 850  | 0.0492        | -               |
| 0.0955 | 900  | 0.0421        | -               |
| 0.1008 | 950  | 0.0384        | -               |
| 0.1061 | 1000 | 0.0415        | -               |
| 0.1114 | 1050 | 0.0399        | -               |
| 0.1167 | 1100 | 0.0393        | -               |
| 0.1221 | 1150 | 0.0397        | -               |
| 0.1274 | 1200 | 0.0405        | -               |
| 0.1327 | 1250 | 0.0324        | -               |
| 0.1380 | 1300 | 0.0318        | -               |
| 0.1433 | 1350 | 0.0296        | -               |
| 0.1486 | 1400 | 0.0315        | -               |
| 0.1539 | 1450 | 0.0309        | -               |
| 0.1592 | 1500 | 0.0301        | -               |
| 0.1645 | 1550 | 0.0316        | -               |
| 0.1698 | 1600 | 0.0315        | -               |
| 0.1751 | 1650 | 0.0301        | -               |
| 0.1804 | 1700 | 0.031         | -               |
| 0.1857 | 1750 | 0.0305        | -               |
| 0.1910 | 1800 | 0.031         | -               |
| 0.1963 | 1850 | 0.029         | -               |
| 0.2017 | 1900 | 0.0317        | -               |
| 0.2070 | 1950 | 0.0276        | -               |
| 0.2123 | 2000 | 0.03          | -               |
| 0.2176 | 2050 | 0.0265        | -               |
| 0.2229 | 2100 | 0.0294        | -               |
| 0.2282 | 2150 | 0.0257        | -               |
| 0.2335 | 2200 | 0.0293        | -               |
| 0.2388 | 2250 | 0.0284        | -               |
| 0.2441 | 2300 | 0.0282        | -               |
| 0.2494 | 2350 | 0.0332        | -               |
| 0.2547 | 2400 | 0.0276        | -               |
| 0.2600 | 2450 | 0.0277        | -               |
| 0.2653 | 2500 | 0.0302        | -               |
| 0.2706 | 2550 | 0.0296        | -               |
| 0.2759 | 2600 | 0.0267        | -               |
| 0.2813 | 2650 | 0.027         | -               |
| 0.2866 | 2700 | 0.0327        | -               |
| 0.2919 | 2750 | 0.0276        | -               |
| 0.2972 | 2800 | 0.0285        | -               |
| 0.3025 | 2850 | 0.0279        | -               |
| 0.3078 | 2900 | 0.0262        | -               |
| 0.3131 | 2950 | 0.0276        | -               |
| 0.3184 | 3000 | 0.0247        | -               |
| 0.3237 | 3050 | 0.0286        | -               |
| 0.3290 | 3100 | 0.026         | -               |
| 0.3343 | 3150 | 0.0279        | -               |
| 0.3396 | 3200 | 0.03          | -               |
| 0.3449 | 3250 | 0.0279        | -               |
| 0.3502 | 3300 | 0.0287        | -               |
| 0.3556 | 3350 | 0.0274        | -               |
| 0.3609 | 3400 | 0.0263        | -               |
| 0.3662 | 3450 | 0.0293        | -               |
| 0.3715 | 3500 | 0.0267        | -               |
| 0.3768 | 3550 | 0.0259        | -               |
| 0.3821 | 3600 | 0.028         | -               |
| 0.3874 | 3650 | 0.0256        | -               |
| 0.3927 | 3700 | 0.0237        | -               |
| 0.3980 | 3750 | 0.0267        | -               |
| 0.4033 | 3800 | 0.0281        | -               |
| 0.4086 | 3850 | 0.0262        | -               |
| 0.4139 | 3900 | 0.0295        | -               |
| 0.4192 | 3950 | 0.0266        | -               |
| 0.4245 | 4000 | 0.024         | -               |
| 0.4298 | 4050 | 0.0284        | -               |
| 0.4352 | 4100 | 0.0275        | -               |
| 0.4405 | 4150 | 0.0265        | -               |
| 0.4458 | 4200 | 0.0267        | -               |
| 0.4511 | 4250 | 0.0267        | -               |
| 0.4564 | 4300 | 0.0265        | -               |
| 0.4617 | 4350 | 0.0252        | -               |
| 0.4670 | 4400 | 0.0258        | -               |
| 0.4723 | 4450 | 0.0265        | -               |
| 0.4776 | 4500 | 0.026         | -               |
| 0.4829 | 4550 | 0.0261        | -               |
| 0.4882 | 4600 | 0.0264        | -               |
| 0.4935 | 4650 | 0.0289        | -               |
| 0.4988 | 4700 | 0.026         | -               |
| 0.5041 | 4750 | 0.0293        | -               |
| 0.5094 | 4800 | 0.0261        | -               |
| 0.5148 | 4850 | 0.0269        | -               |
| 0.5201 | 4900 | 0.027         | -               |
| 0.5254 | 4950 | 0.026         | -               |
| 0.5307 | 5000 | 0.0294        | -               |
| 0.5360 | 5050 | 0.0257        | -               |
| 0.5413 | 5100 | 0.0288        | -               |
| 0.5466 | 5150 | 0.029         | -               |
| 0.5519 | 5200 | 0.027         | -               |
| 0.5572 | 5250 | 0.0272        | -               |
| 0.5625 | 5300 | 0.0265        | -               |
| 0.5678 | 5350 | 0.0282        | -               |
| 0.5731 | 5400 | 0.0264        | -               |
| 0.5784 | 5450 | 0.0279        | -               |
| 0.5837 | 5500 | 0.0262        | -               |
| 0.5890 | 5550 | 0.0262        | -               |
| 0.5944 | 5600 | 0.0291        | -               |
| 0.5997 | 5650 | 0.0251        | -               |
| 0.6050 | 5700 | 0.0271        | -               |
| 0.6103 | 5750 | 0.0276        | -               |
| 0.6156 | 5800 | 0.0267        | -               |
| 0.6209 | 5850 | 0.0254        | -               |
| 0.6262 | 5900 | 0.0287        | -               |
| 0.6315 | 5950 | 0.0268        | -               |
| 0.6368 | 6000 | 0.0273        | -               |
| 0.6421 | 6050 | 0.0243        | -               |
| 0.6474 | 6100 | 0.0268        | -               |
| 0.6527 | 6150 | 0.0252        | -               |
| 0.6580 | 6200 | 0.0243        | -               |
| 0.6633 | 6250 | 0.0258        | -               |
| 0.6686 | 6300 | 0.0263        | -               |
| 0.6740 | 6350 | 0.0278        | -               |
| 0.6793 | 6400 | 0.0265        | -               |
| 0.6846 | 6450 | 0.0265        | -               |
| 0.6899 | 6500 | 0.0263        | -               |
| 0.6952 | 6550 | 0.0243        | -               |
| 0.7005 | 6600 | 0.0256        | -               |
| 0.7058 | 6650 | 0.0242        | -               |
| 0.7111 | 6700 | 0.0247        | -               |
| 0.7164 | 6750 | 0.0257        | -               |
| 0.7217 | 6800 | 0.0248        | -               |
| 0.7270 | 6850 | 0.0274        | -               |
| 0.7323 | 6900 | 0.025         | -               |
| 0.7376 | 6950 | 0.0249        | -               |
| 0.7429 | 7000 | 0.0263        | -               |
| 0.7482 | 7050 | 0.0257        | -               |
| 0.7536 | 7100 | 0.0259        | -               |
| 0.7589 | 7150 | 0.0252        | -               |
| 0.7642 | 7200 | 0.0227        | -               |
| 0.7695 | 7250 | 0.0275        | -               |
| 0.7748 | 7300 | 0.0258        | -               |
| 0.7801 | 7350 | 0.0264        | -               |
| 0.7854 | 7400 | 0.0248        | -               |
| 0.7907 | 7450 | 0.0233        | -               |
| 0.7960 | 7500 | 0.0241        | -               |
| 0.8013 | 7550 | 0.0262        | -               |
| 0.8066 | 7600 | 0.0237        | -               |
| 0.8119 | 7650 | 0.028         | -               |
| 0.8172 | 7700 | 0.0226        | -               |
| 0.8225 | 7750 | 0.0261        | -               |
| 0.8278 | 7800 | 0.0243        | -               |
| 0.8332 | 7850 | 0.0252        | -               |
| 0.8385 | 7900 | 0.0249        | -               |
| 0.8438 | 7950 | 0.0248        | -               |
| 0.8491 | 8000 | 0.0274        | -               |
| 0.8544 | 8050 | 0.0255        | -               |
| 0.8597 | 8100 | 0.0255        | -               |
| 0.8650 | 8150 | 0.0246        | -               |
| 0.8703 | 8200 | 0.0257        | -               |
| 0.8756 | 8250 | 0.0217        | -               |
| 0.8809 | 8300 | 0.0247        | -               |
| 0.8862 | 8350 | 0.0264        | -               |
| 0.8915 | 8400 | 0.0268        | -               |
| 0.8968 | 8450 | 0.0254        | -               |
| 0.9021 | 8500 | 0.0233        | -               |
| 0.9075 | 8550 | 0.0248        | -               |
| 0.9128 | 8600 | 0.0273        | -               |
| 0.9181 | 8650 | 0.0244        | -               |
| 0.9234 | 8700 | 0.0263        | -               |
| 0.9287 | 8750 | 0.0251        | -               |
| 0.9340 | 8800 | 0.0277        | -               |
| 0.9393 | 8850 | 0.0227        | -               |
| 0.9446 | 8900 | 0.0217        | -               |
| 0.9499 | 8950 | 0.0267        | -               |
| 0.9552 | 9000 | 0.0235        | -               |
| 0.9605 | 9050 | 0.0252        | -               |
| 0.9658 | 9100 | 0.0263        | -               |
| 0.9711 | 9150 | 0.0252        | -               |
| 0.9764 | 9200 | 0.0235        | -               |
| 0.9817 | 9250 | 0.0254        | -               |
| 0.9871 | 9300 | 0.0254        | -               |
| 0.9924 | 9350 | 0.0271        | -               |
| 0.9977 | 9400 | 0.0265        | -               |

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