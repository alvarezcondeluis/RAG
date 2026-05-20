---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Show Vanguard funds and the CEOs of companies in their portfolios
- text: How many holdings does VOO have?
- text: What is quantum physics?
- text: Which funds have more than 30% in Consumer Discretionary sector?
- text: Find funds with more than 15% allocation to Europe
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
| Word count   | 3   | 7.9342 | 16  |

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
| 0.0001 | 1    | 0.2617        | -               |
| 0.0053 | 50   | 0.2475        | -               |
| 0.0107 | 100  | 0.212         | -               |
| 0.0160 | 150  | 0.2053        | -               |
| 0.0214 | 200  | 0.1822        | -               |
| 0.0267 | 250  | 0.1797        | -               |
| 0.0321 | 300  | 0.1658        | -               |
| 0.0374 | 350  | 0.1251        | -               |
| 0.0428 | 400  | 0.11          | -               |
| 0.0481 | 450  | 0.0954        | -               |
| 0.0534 | 500  | 0.0783        | -               |
| 0.0588 | 550  | 0.0712        | -               |
| 0.0641 | 600  | 0.0577        | -               |
| 0.0695 | 650  | 0.0525        | -               |
| 0.0748 | 700  | 0.0536        | -               |
| 0.0802 | 750  | 0.052         | -               |
| 0.0855 | 800  | 0.0434        | -               |
| 0.0909 | 850  | 0.0435        | -               |
| 0.0962 | 900  | 0.0467        | -               |
| 0.1015 | 950  | 0.0477        | -               |
| 0.1069 | 1000 | 0.0407        | -               |
| 0.1122 | 1050 | 0.0381        | -               |
| 0.1176 | 1100 | 0.0417        | -               |
| 0.1229 | 1150 | 0.0369        | -               |
| 0.1283 | 1200 | 0.0348        | -               |
| 0.1336 | 1250 | 0.0298        | -               |
| 0.1390 | 1300 | 0.0356        | -               |
| 0.1443 | 1350 | 0.0373        | -               |
| 0.1497 | 1400 | 0.0305        | -               |
| 0.1550 | 1450 | 0.0303        | -               |
| 0.1603 | 1500 | 0.0328        | -               |
| 0.1657 | 1550 | 0.035         | -               |
| 0.1710 | 1600 | 0.033         | -               |
| 0.1764 | 1650 | 0.0269        | -               |
| 0.1817 | 1700 | 0.0338        | -               |
| 0.1871 | 1750 | 0.0288        | -               |
| 0.1924 | 1800 | 0.0306        | -               |
| 0.1978 | 1850 | 0.0299        | -               |
| 0.2031 | 1900 | 0.0299        | -               |
| 0.2084 | 1950 | 0.029         | -               |
| 0.2138 | 2000 | 0.0311        | -               |
| 0.2191 | 2050 | 0.0293        | -               |
| 0.2245 | 2100 | 0.0306        | -               |
| 0.2298 | 2150 | 0.0287        | -               |
| 0.2352 | 2200 | 0.0276        | -               |
| 0.2405 | 2250 | 0.0303        | -               |
| 0.2459 | 2300 | 0.0311        | -               |
| 0.2512 | 2350 | 0.0301        | -               |
| 0.2565 | 2400 | 0.0273        | -               |
| 0.2619 | 2450 | 0.0321        | -               |
| 0.2672 | 2500 | 0.0314        | -               |
| 0.2726 | 2550 | 0.0283        | -               |
| 0.2779 | 2600 | 0.0279        | -               |
| 0.2833 | 2650 | 0.0296        | -               |
| 0.2886 | 2700 | 0.0287        | -               |
| 0.2940 | 2750 | 0.0299        | -               |
| 0.2993 | 2800 | 0.0287        | -               |
| 0.3046 | 2850 | 0.0299        | -               |
| 0.3100 | 2900 | 0.0292        | -               |
| 0.3153 | 2950 | 0.0252        | -               |
| 0.3207 | 3000 | 0.0294        | -               |
| 0.3260 | 3050 | 0.0291        | -               |
| 0.3314 | 3100 | 0.031         | -               |
| 0.3367 | 3150 | 0.0311        | -               |
| 0.3421 | 3200 | 0.0296        | -               |
| 0.3474 | 3250 | 0.0326        | -               |
| 0.3528 | 3300 | 0.0287        | -               |
| 0.3581 | 3350 | 0.0259        | -               |
| 0.3634 | 3400 | 0.0317        | -               |
| 0.3688 | 3450 | 0.0273        | -               |
| 0.3741 | 3500 | 0.0309        | -               |
| 0.3795 | 3550 | 0.0291        | -               |
| 0.3848 | 3600 | 0.0302        | -               |
| 0.3902 | 3650 | 0.0256        | -               |
| 0.3955 | 3700 | 0.0296        | -               |
| 0.4009 | 3750 | 0.026         | -               |
| 0.4062 | 3800 | 0.0277        | -               |
| 0.4115 | 3850 | 0.0283        | -               |
| 0.4169 | 3900 | 0.0266        | -               |
| 0.4222 | 3950 | 0.0275        | -               |
| 0.4276 | 4000 | 0.0295        | -               |
| 0.4329 | 4050 | 0.0306        | -               |
| 0.4383 | 4100 | 0.0289        | -               |
| 0.4436 | 4150 | 0.0297        | -               |
| 0.4490 | 4200 | 0.0272        | -               |
| 0.4543 | 4250 | 0.0259        | -               |
| 0.4596 | 4300 | 0.0264        | -               |
| 0.4650 | 4350 | 0.0273        | -               |
| 0.4703 | 4400 | 0.0291        | -               |
| 0.4757 | 4450 | 0.0293        | -               |
| 0.4810 | 4500 | 0.0267        | -               |
| 0.4864 | 4550 | 0.029         | -               |
| 0.4917 | 4600 | 0.0261        | -               |
| 0.4971 | 4650 | 0.027         | -               |
| 0.5024 | 4700 | 0.0256        | -               |
| 0.5077 | 4750 | 0.0311        | -               |
| 0.5131 | 4800 | 0.0288        | -               |
| 0.5184 | 4850 | 0.0275        | -               |
| 0.5238 | 4900 | 0.0266        | -               |
| 0.5291 | 4950 | 0.0321        | -               |
| 0.5345 | 5000 | 0.0288        | -               |
| 0.5398 | 5050 | 0.0266        | -               |
| 0.5452 | 5100 | 0.0287        | -               |
| 0.5505 | 5150 | 0.0257        | -               |
| 0.5559 | 5200 | 0.0326        | -               |
| 0.5612 | 5250 | 0.0276        | -               |
| 0.5665 | 5300 | 0.0263        | -               |
| 0.5719 | 5350 | 0.0284        | -               |
| 0.5772 | 5400 | 0.0268        | -               |
| 0.5826 | 5450 | 0.0297        | -               |
| 0.5879 | 5500 | 0.0283        | -               |
| 0.5933 | 5550 | 0.0275        | -               |
| 0.5986 | 5600 | 0.0256        | -               |
| 0.6040 | 5650 | 0.0282        | -               |
| 0.6093 | 5700 | 0.0272        | -               |
| 0.6146 | 5750 | 0.0283        | -               |
| 0.6200 | 5800 | 0.0278        | -               |
| 0.6253 | 5850 | 0.029         | -               |
| 0.6307 | 5900 | 0.0281        | -               |
| 0.6360 | 5950 | 0.0268        | -               |
| 0.6414 | 6000 | 0.0272        | -               |
| 0.6467 | 6050 | 0.0278        | -               |
| 0.6521 | 6100 | 0.0266        | -               |
| 0.6574 | 6150 | 0.0245        | -               |
| 0.6627 | 6200 | 0.028         | -               |
| 0.6681 | 6250 | 0.0267        | -               |
| 0.6734 | 6300 | 0.0273        | -               |
| 0.6788 | 6350 | 0.0271        | -               |
| 0.6841 | 6400 | 0.0276        | -               |
| 0.6895 | 6450 | 0.0272        | -               |
| 0.6948 | 6500 | 0.0299        | -               |
| 0.7002 | 6550 | 0.0285        | -               |
| 0.7055 | 6600 | 0.0268        | -               |
| 0.7108 | 6650 | 0.0283        | -               |
| 0.7162 | 6700 | 0.0253        | -               |
| 0.7215 | 6750 | 0.0272        | -               |
| 0.7269 | 6800 | 0.0243        | -               |
| 0.7322 | 6850 | 0.0307        | -               |
| 0.7376 | 6900 | 0.0264        | -               |
| 0.7429 | 6950 | 0.0281        | -               |
| 0.7483 | 7000 | 0.0266        | -               |
| 0.7536 | 7050 | 0.0289        | -               |
| 0.7590 | 7100 | 0.0277        | -               |
| 0.7643 | 7150 | 0.0277        | -               |
| 0.7696 | 7200 | 0.0295        | -               |
| 0.7750 | 7250 | 0.0274        | -               |
| 0.7803 | 7300 | 0.0243        | -               |
| 0.7857 | 7350 | 0.0245        | -               |
| 0.7910 | 7400 | 0.0266        | -               |
| 0.7964 | 7450 | 0.0264        | -               |
| 0.8017 | 7500 | 0.0275        | -               |
| 0.8071 | 7550 | 0.0269        | -               |
| 0.8124 | 7600 | 0.0294        | -               |
| 0.8177 | 7650 | 0.0247        | -               |
| 0.8231 | 7700 | 0.0278        | -               |
| 0.8284 | 7750 | 0.025         | -               |
| 0.8338 | 7800 | 0.028         | -               |
| 0.8391 | 7850 | 0.0274        | -               |
| 0.8445 | 7900 | 0.0298        | -               |
| 0.8498 | 7950 | 0.0262        | -               |
| 0.8552 | 8000 | 0.025         | -               |
| 0.8605 | 8050 | 0.0243        | -               |
| 0.8658 | 8100 | 0.0299        | -               |
| 0.8712 | 8150 | 0.0272        | -               |
| 0.8765 | 8200 | 0.0261        | -               |
| 0.8819 | 8250 | 0.0269        | -               |
| 0.8872 | 8300 | 0.0247        | -               |
| 0.8926 | 8350 | 0.0247        | -               |
| 0.8979 | 8400 | 0.0292        | -               |
| 0.9033 | 8450 | 0.0292        | -               |
| 0.9086 | 8500 | 0.0281        | -               |
| 0.9139 | 8550 | 0.0278        | -               |
| 0.9193 | 8600 | 0.0297        | -               |
| 0.9246 | 8650 | 0.028         | -               |
| 0.9300 | 8700 | 0.0273        | -               |
| 0.9353 | 8750 | 0.0284        | -               |
| 0.9407 | 8800 | 0.0276        | -               |
| 0.9460 | 8850 | 0.025         | -               |
| 0.9514 | 8900 | 0.0247        | -               |
| 0.9567 | 8950 | 0.0256        | -               |
| 0.9621 | 9000 | 0.0272        | -               |
| 0.9674 | 9050 | 0.0271        | -               |
| 0.9727 | 9100 | 0.0269        | -               |
| 0.9781 | 9150 | 0.0264        | -               |
| 0.9834 | 9200 | 0.0254        | -               |
| 0.9888 | 9250 | 0.0261        | -               |
| 0.9941 | 9300 | 0.0264        | -               |
| 0.9995 | 9350 | 0.0262        | -               |

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