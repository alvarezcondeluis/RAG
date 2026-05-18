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
| 0.0001 | 1    | 0.2896        | -               |
| 0.0054 | 50   | 0.2448        | -               |
| 0.0107 | 100  | 0.2152        | -               |
| 0.0161 | 150  | 0.2073        | -               |
| 0.0214 | 200  | 0.1889        | -               |
| 0.0268 | 250  | 0.1734        | -               |
| 0.0321 | 300  | 0.1496        | -               |
| 0.0375 | 350  | 0.1373        | -               |
| 0.0429 | 400  | 0.1163        | -               |
| 0.0482 | 450  | 0.1011        | -               |
| 0.0536 | 500  | 0.0852        | -               |
| 0.0589 | 550  | 0.0684        | -               |
| 0.0643 | 600  | 0.0643        | -               |
| 0.0696 | 650  | 0.0559        | -               |
| 0.0750 | 700  | 0.0627        | -               |
| 0.0804 | 750  | 0.0532        | -               |
| 0.0857 | 800  | 0.0479        | -               |
| 0.0911 | 850  | 0.0553        | -               |
| 0.0964 | 900  | 0.0493        | -               |
| 0.1018 | 950  | 0.0469        | -               |
| 0.1071 | 1000 | 0.0454        | -               |
| 0.1125 | 1050 | 0.0404        | -               |
| 0.1179 | 1100 | 0.0386        | -               |
| 0.1232 | 1150 | 0.0356        | -               |
| 0.1286 | 1200 | 0.0407        | -               |
| 0.1339 | 1250 | 0.0371        | -               |
| 0.1393 | 1300 | 0.0372        | -               |
| 0.1446 | 1350 | 0.035         | -               |
| 0.1500 | 1400 | 0.0336        | -               |
| 0.1554 | 1450 | 0.0362        | -               |
| 0.1607 | 1500 | 0.0357        | -               |
| 0.1661 | 1550 | 0.0336        | -               |
| 0.1714 | 1600 | 0.0365        | -               |
| 0.1768 | 1650 | 0.0286        | -               |
| 0.1821 | 1700 | 0.0305        | -               |
| 0.1875 | 1750 | 0.0329        | -               |
| 0.1929 | 1800 | 0.0325        | -               |
| 0.1982 | 1850 | 0.0299        | -               |
| 0.2036 | 1900 | 0.0293        | -               |
| 0.2089 | 1950 | 0.0288        | -               |
| 0.2143 | 2000 | 0.0317        | -               |
| 0.2197 | 2050 | 0.028         | -               |
| 0.2250 | 2100 | 0.0327        | -               |
| 0.2304 | 2150 | 0.0308        | -               |
| 0.2357 | 2200 | 0.0299        | -               |
| 0.2411 | 2250 | 0.0294        | -               |
| 0.2464 | 2300 | 0.0302        | -               |
| 0.2518 | 2350 | 0.0288        | -               |
| 0.2572 | 2400 | 0.0316        | -               |
| 0.2625 | 2450 | 0.0333        | -               |
| 0.2679 | 2500 | 0.032         | -               |
| 0.2732 | 2550 | 0.0293        | -               |
| 0.2786 | 2600 | 0.0285        | -               |
| 0.2839 | 2650 | 0.027         | -               |
| 0.2893 | 2700 | 0.0302        | -               |
| 0.2947 | 2750 | 0.0296        | -               |
| 0.3000 | 2800 | 0.0292        | -               |
| 0.3054 | 2850 | 0.0279        | -               |
| 0.3107 | 2900 | 0.0284        | -               |
| 0.3161 | 2950 | 0.0271        | -               |
| 0.3214 | 3000 | 0.0322        | -               |
| 0.3268 | 3050 | 0.0292        | -               |
| 0.3322 | 3100 | 0.0272        | -               |
| 0.3375 | 3150 | 0.0288        | -               |
| 0.3429 | 3200 | 0.0272        | -               |
| 0.3482 | 3250 | 0.0283        | -               |
| 0.3536 | 3300 | 0.0304        | -               |
| 0.3589 | 3350 | 0.0288        | -               |
| 0.3643 | 3400 | 0.0296        | -               |
| 0.3697 | 3450 | 0.0298        | -               |
| 0.3750 | 3500 | 0.03          | -               |
| 0.3804 | 3550 | 0.0263        | -               |
| 0.3857 | 3600 | 0.0281        | -               |
| 0.3911 | 3650 | 0.0302        | -               |
| 0.3964 | 3700 | 0.028         | -               |
| 0.4018 | 3750 | 0.0296        | -               |
| 0.4072 | 3800 | 0.0299        | -               |
| 0.4125 | 3850 | 0.0272        | -               |
| 0.4179 | 3900 | 0.0282        | -               |
| 0.4232 | 3950 | 0.0309        | -               |
| 0.4286 | 4000 | 0.0293        | -               |
| 0.4339 | 4050 | 0.029         | -               |
| 0.4393 | 4100 | 0.0304        | -               |
| 0.4447 | 4150 | 0.0268        | -               |
| 0.4500 | 4200 | 0.0331        | -               |
| 0.4554 | 4250 | 0.0259        | -               |
| 0.4607 | 4300 | 0.0298        | -               |
| 0.4661 | 4350 | 0.0275        | -               |
| 0.4714 | 4400 | 0.026         | -               |
| 0.4768 | 4450 | 0.0294        | -               |
| 0.4822 | 4500 | 0.0285        | -               |
| 0.4875 | 4550 | 0.0283        | -               |
| 0.4929 | 4600 | 0.0272        | -               |
| 0.4982 | 4650 | 0.0281        | -               |
| 0.5036 | 4700 | 0.026         | -               |
| 0.5089 | 4750 | 0.0292        | -               |
| 0.5143 | 4800 | 0.0291        | -               |
| 0.5197 | 4850 | 0.0315        | -               |
| 0.5250 | 4900 | 0.0277        | -               |
| 0.5304 | 4950 | 0.0273        | -               |
| 0.5357 | 5000 | 0.0259        | -               |
| 0.5411 | 5050 | 0.0292        | -               |
| 0.5464 | 5100 | 0.0291        | -               |
| 0.5518 | 5150 | 0.0315        | -               |
| 0.5572 | 5200 | 0.0274        | -               |
| 0.5625 | 5250 | 0.0306        | -               |
| 0.5679 | 5300 | 0.0303        | -               |
| 0.5732 | 5350 | 0.0273        | -               |
| 0.5786 | 5400 | 0.0298        | -               |
| 0.5839 | 5450 | 0.0274        | -               |
| 0.5893 | 5500 | 0.0275        | -               |
| 0.5947 | 5550 | 0.0278        | -               |
| 0.6000 | 5600 | 0.0312        | -               |
| 0.6054 | 5650 | 0.0324        | -               |
| 0.6107 | 5700 | 0.027         | -               |
| 0.6161 | 5750 | 0.0277        | -               |
| 0.6215 | 5800 | 0.0281        | -               |
| 0.6268 | 5850 | 0.03          | -               |
| 0.6322 | 5900 | 0.0275        | -               |
| 0.6375 | 5950 | 0.0258        | -               |
| 0.6429 | 6000 | 0.0282        | -               |
| 0.6482 | 6050 | 0.0263        | -               |
| 0.6536 | 6100 | 0.0278        | -               |
| 0.6590 | 6150 | 0.0248        | -               |
| 0.6643 | 6200 | 0.0276        | -               |
| 0.6697 | 6250 | 0.0275        | -               |
| 0.6750 | 6300 | 0.0295        | -               |
| 0.6804 | 6350 | 0.0265        | -               |
| 0.6857 | 6400 | 0.0279        | -               |
| 0.6911 | 6450 | 0.026         | -               |
| 0.6965 | 6500 | 0.0284        | -               |
| 0.7018 | 6550 | 0.0262        | -               |
| 0.7072 | 6600 | 0.0273        | -               |
| 0.7125 | 6650 | 0.0296        | -               |
| 0.7179 | 6700 | 0.026         | -               |
| 0.7232 | 6750 | 0.0312        | -               |
| 0.7286 | 6800 | 0.0263        | -               |
| 0.7340 | 6850 | 0.0296        | -               |
| 0.7393 | 6900 | 0.029         | -               |
| 0.7447 | 6950 | 0.0296        | -               |
| 0.7500 | 7000 | 0.0243        | -               |
| 0.7554 | 7050 | 0.0278        | -               |
| 0.7607 | 7100 | 0.0269        | -               |
| 0.7661 | 7150 | 0.0284        | -               |
| 0.7715 | 7200 | 0.0261        | -               |
| 0.7768 | 7250 | 0.0251        | -               |
| 0.7822 | 7300 | 0.0307        | -               |
| 0.7875 | 7350 | 0.0229        | -               |
| 0.7929 | 7400 | 0.0261        | -               |
| 0.7982 | 7450 | 0.0259        | -               |
| 0.8036 | 7500 | 0.0273        | -               |
| 0.8090 | 7550 | 0.0288        | -               |
| 0.8143 | 7600 | 0.0278        | -               |
| 0.8197 | 7650 | 0.0286        | -               |
| 0.8250 | 7700 | 0.0279        | -               |
| 0.8304 | 7750 | 0.0254        | -               |
| 0.8357 | 7800 | 0.0289        | -               |
| 0.8411 | 7850 | 0.0268        | -               |
| 0.8465 | 7900 | 0.025         | -               |
| 0.8518 | 7950 | 0.0292        | -               |
| 0.8572 | 8000 | 0.0267        | -               |
| 0.8625 | 8050 | 0.0255        | -               |
| 0.8679 | 8100 | 0.0266        | -               |
| 0.8732 | 8150 | 0.0256        | -               |
| 0.8786 | 8200 | 0.0271        | -               |
| 0.8840 | 8250 | 0.0278        | -               |
| 0.8893 | 8300 | 0.0282        | -               |
| 0.8947 | 8350 | 0.0299        | -               |
| 0.9000 | 8400 | 0.0261        | -               |
| 0.9054 | 8450 | 0.0272        | -               |
| 0.9107 | 8500 | 0.0293        | -               |
| 0.9161 | 8550 | 0.0251        | -               |
| 0.9215 | 8600 | 0.0269        | -               |
| 0.9268 | 8650 | 0.0308        | -               |
| 0.9322 | 8700 | 0.0309        | -               |
| 0.9375 | 8750 | 0.0267        | -               |
| 0.9429 | 8800 | 0.0284        | -               |
| 0.9482 | 8850 | 0.0287        | -               |
| 0.9536 | 8900 | 0.0265        | -               |
| 0.9590 | 8950 | 0.0256        | -               |
| 0.9643 | 9000 | 0.0305        | -               |
| 0.9697 | 9050 | 0.0257        | -               |
| 0.9750 | 9100 | 0.027         | -               |
| 0.9804 | 9150 | 0.0295        | -               |
| 0.9857 | 9200 | 0.0268        | -               |
| 0.9911 | 9250 | 0.0274        | -               |
| 0.9965 | 9300 | 0.0294        | -               |

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