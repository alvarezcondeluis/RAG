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
- text: Which year had the worst total return for VOO?
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
| Word count   | 3   | 7.5753 | 16  |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (3, 3)
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
| Epoch  | Step  | Training Loss | Validation Loss |
|:------:|:-----:|:-------------:|:---------------:|
| 0.0002 | 1     | 0.2552        | -               |
| 0.0097 | 50    | 0.2384        | -               |
| 0.0193 | 100   | 0.2125        | -               |
| 0.0290 | 150   | 0.2081        | -               |
| 0.0387 | 200   | 0.1977        | -               |
| 0.0484 | 250   | 0.1862        | -               |
| 0.0580 | 300   | 0.1677        | -               |
| 0.0677 | 350   | 0.149         | -               |
| 0.0774 | 400   | 0.1256        | -               |
| 0.0870 | 450   | 0.1171        | -               |
| 0.0967 | 500   | 0.0952        | -               |
| 0.1064 | 550   | 0.0687        | -               |
| 0.1161 | 600   | 0.0549        | -               |
| 0.1257 | 650   | 0.0467        | -               |
| 0.1354 | 700   | 0.0504        | -               |
| 0.1451 | 750   | 0.038         | -               |
| 0.1547 | 800   | 0.0339        | -               |
| 0.1644 | 850   | 0.0407        | -               |
| 0.1741 | 900   | 0.0312        | -               |
| 0.1838 | 950   | 0.0299        | -               |
| 0.1934 | 1000  | 0.0356        | -               |
| 0.2031 | 1050  | 0.0271        | -               |
| 0.2128 | 1100  | 0.031         | -               |
| 0.2224 | 1150  | 0.0284        | -               |
| 0.2321 | 1200  | 0.0282        | -               |
| 0.2418 | 1250  | 0.0228        | -               |
| 0.2515 | 1300  | 0.0289        | -               |
| 0.2611 | 1350  | 0.025         | -               |
| 0.2708 | 1400  | 0.0243        | -               |
| 0.2805 | 1450  | 0.026         | -               |
| 0.2901 | 1500  | 0.0258        | -               |
| 0.2998 | 1550  | 0.0222        | -               |
| 0.3095 | 1600  | 0.0245        | -               |
| 0.3191 | 1650  | 0.022         | -               |
| 0.3288 | 1700  | 0.0211        | -               |
| 0.3385 | 1750  | 0.0233        | -               |
| 0.3482 | 1800  | 0.0196        | -               |
| 0.3578 | 1850  | 0.0232        | -               |
| 0.3675 | 1900  | 0.0232        | -               |
| 0.3772 | 1950  | 0.0187        | -               |
| 0.3868 | 2000  | 0.02          | -               |
| 0.3965 | 2050  | 0.0204        | -               |
| 0.4062 | 2100  | 0.0234        | -               |
| 0.4159 | 2150  | 0.0204        | -               |
| 0.4255 | 2200  | 0.0185        | -               |
| 0.4352 | 2250  | 0.0192        | -               |
| 0.4449 | 2300  | 0.02          | -               |
| 0.4545 | 2350  | 0.0194        | -               |
| 0.4642 | 2400  | 0.0209        | -               |
| 0.4739 | 2450  | 0.0192        | -               |
| 0.4836 | 2500  | 0.0228        | -               |
| 0.4932 | 2550  | 0.0193        | -               |
| 0.5029 | 2600  | 0.0202        | -               |
| 0.5126 | 2650  | 0.0201        | -               |
| 0.5222 | 2700  | 0.0178        | -               |
| 0.5319 | 2750  | 0.0209        | -               |
| 0.5416 | 2800  | 0.0211        | -               |
| 0.5513 | 2850  | 0.0207        | -               |
| 0.5609 | 2900  | 0.0231        | -               |
| 0.5706 | 2950  | 0.0194        | -               |
| 0.5803 | 3000  | 0.0192        | -               |
| 0.5899 | 3050  | 0.0212        | -               |
| 0.5996 | 3100  | 0.0206        | -               |
| 0.6093 | 3150  | 0.0183        | -               |
| 0.6190 | 3200  | 0.0177        | -               |
| 0.6286 | 3250  | 0.0172        | -               |
| 0.6383 | 3300  | 0.017         | -               |
| 0.6480 | 3350  | 0.0206        | -               |
| 0.6576 | 3400  | 0.0192        | -               |
| 0.6673 | 3450  | 0.0219        | -               |
| 0.6770 | 3500  | 0.0193        | -               |
| 0.6867 | 3550  | 0.02          | -               |
| 0.6963 | 3600  | 0.018         | -               |
| 0.7060 | 3650  | 0.0193        | -               |
| 0.7157 | 3700  | 0.018         | -               |
| 0.7253 | 3750  | 0.0183        | -               |
| 0.7350 | 3800  | 0.0222        | -               |
| 0.7447 | 3850  | 0.0198        | -               |
| 0.7544 | 3900  | 0.0196        | -               |
| 0.7640 | 3950  | 0.0205        | -               |
| 0.7737 | 4000  | 0.0168        | -               |
| 0.7834 | 4050  | 0.0207        | -               |
| 0.7930 | 4100  | 0.018         | -               |
| 0.8027 | 4150  | 0.0201        | -               |
| 0.8124 | 4200  | 0.0172        | -               |
| 0.8221 | 4250  | 0.0187        | -               |
| 0.8317 | 4300  | 0.0191        | -               |
| 0.8414 | 4350  | 0.0194        | -               |
| 0.8511 | 4400  | 0.0171        | -               |
| 0.8607 | 4450  | 0.0179        | -               |
| 0.8704 | 4500  | 0.018         | -               |
| 0.8801 | 4550  | 0.0192        | -               |
| 0.8897 | 4600  | 0.0215        | -               |
| 0.8994 | 4650  | 0.0183        | -               |
| 0.9091 | 4700  | 0.019         | -               |
| 0.9188 | 4750  | 0.0182        | -               |
| 0.9284 | 4800  | 0.0212        | -               |
| 0.9381 | 4850  | 0.0176        | -               |
| 0.9478 | 4900  | 0.0176        | -               |
| 0.9574 | 4950  | 0.0195        | -               |
| 0.9671 | 5000  | 0.0177        | -               |
| 0.9768 | 5050  | 0.019         | -               |
| 0.9865 | 5100  | 0.018         | -               |
| 0.9961 | 5150  | 0.0195        | -               |
| 1.0058 | 5200  | 0.0173        | -               |
| 1.0155 | 5250  | 0.0208        | -               |
| 1.0251 | 5300  | 0.0204        | -               |
| 1.0348 | 5350  | 0.0188        | -               |
| 1.0445 | 5400  | 0.0177        | -               |
| 1.0542 | 5450  | 0.0178        | -               |
| 1.0638 | 5500  | 0.0181        | -               |
| 1.0735 | 5550  | 0.0187        | -               |
| 1.0832 | 5600  | 0.018         | -               |
| 1.0928 | 5650  | 0.0216        | -               |
| 1.1025 | 5700  | 0.0179        | -               |
| 1.1122 | 5750  | 0.0188        | -               |
| 1.1219 | 5800  | 0.0212        | -               |
| 1.1315 | 5850  | 0.018         | -               |
| 1.1412 | 5900  | 0.0189        | -               |
| 1.1509 | 5950  | 0.0289        | -               |
| 1.1605 | 6000  | 0.0208        | -               |
| 1.1702 | 6050  | 0.0209        | -               |
| 1.1799 | 6100  | 0.0182        | -               |
| 1.1896 | 6150  | 0.0232        | -               |
| 1.1992 | 6200  | 0.0189        | -               |
| 1.2089 | 6250  | 0.0231        | -               |
| 1.2186 | 6300  | 0.0186        | -               |
| 1.2282 | 6350  | 0.0179        | -               |
| 1.2379 | 6400  | 0.0167        | -               |
| 1.2476 | 6450  | 0.0186        | -               |
| 1.2573 | 6500  | 0.0162        | -               |
| 1.2669 | 6550  | 0.0206        | -               |
| 1.2766 | 6600  | 0.0164        | -               |
| 1.2863 | 6650  | 0.0195        | -               |
| 1.2959 | 6700  | 0.0187        | -               |
| 1.3056 | 6750  | 0.0185        | -               |
| 1.3153 | 6800  | 0.0169        | -               |
| 1.3250 | 6850  | 0.0186        | -               |
| 1.3346 | 6900  | 0.0163        | -               |
| 1.3443 | 6950  | 0.0182        | -               |
| 1.3540 | 7000  | 0.0188        | -               |
| 1.3636 | 7050  | 0.0203        | -               |
| 1.3733 | 7100  | 0.0187        | -               |
| 1.3830 | 7150  | 0.0185        | -               |
| 1.3926 | 7200  | 0.0181        | -               |
| 1.4023 | 7250  | 0.0188        | -               |
| 1.4120 | 7300  | 0.0174        | -               |
| 1.4217 | 7350  | 0.0168        | -               |
| 1.4313 | 7400  | 0.0187        | -               |
| 1.4410 | 7450  | 0.0172        | -               |
| 1.4507 | 7500  | 0.0182        | -               |
| 1.4603 | 7550  | 0.0196        | -               |
| 1.4700 | 7600  | 0.0162        | -               |
| 1.4797 | 7650  | 0.0172        | -               |
| 1.4894 | 7700  | 0.0205        | -               |
| 1.4990 | 7750  | 0.0189        | -               |
| 1.5087 | 7800  | 0.0199        | -               |
| 1.5184 | 7850  | 0.0176        | -               |
| 1.5280 | 7900  | 0.0192        | -               |
| 1.5377 | 7950  | 0.0207        | -               |
| 1.5474 | 8000  | 0.0178        | -               |
| 1.5571 | 8050  | 0.0181        | -               |
| 1.5667 | 8100  | 0.0195        | -               |
| 1.5764 | 8150  | 0.0194        | -               |
| 1.5861 | 8200  | 0.0196        | -               |
| 1.5957 | 8250  | 0.0169        | -               |
| 1.6054 | 8300  | 0.0178        | -               |
| 1.6151 | 8350  | 0.0178        | -               |
| 1.6248 | 8400  | 0.0166        | -               |
| 1.6344 | 8450  | 0.0174        | -               |
| 1.6441 | 8500  | 0.0197        | -               |
| 1.6538 | 8550  | 0.0196        | -               |
| 1.6634 | 8600  | 0.0206        | -               |
| 1.6731 | 8650  | 0.0189        | -               |
| 1.6828 | 8700  | 0.0181        | -               |
| 1.6925 | 8750  | 0.0188        | -               |
| 1.7021 | 8800  | 0.0221        | -               |
| 1.7118 | 8850  | 0.0234        | -               |
| 1.7215 | 8900  | 0.0208        | -               |
| 1.7311 | 8950  | 0.0199        | -               |
| 1.7408 | 9000  | 0.0194        | -               |
| 1.7505 | 9050  | 0.0189        | -               |
| 1.7602 | 9100  | 0.0213        | -               |
| 1.7698 | 9150  | 0.0187        | -               |
| 1.7795 | 9200  | 0.0194        | -               |
| 1.7892 | 9250  | 0.0185        | -               |
| 1.7988 | 9300  | 0.0222        | -               |
| 1.8085 | 9350  | 0.0189        | -               |
| 1.8182 | 9400  | 0.0207        | -               |
| 1.8279 | 9450  | 0.0171        | -               |
| 1.8375 | 9500  | 0.0171        | -               |
| 1.8472 | 9550  | 0.017         | -               |
| 1.8569 | 9600  | 0.0196        | -               |
| 1.8665 | 9650  | 0.0183        | -               |
| 1.8762 | 9700  | 0.0184        | -               |
| 1.8859 | 9750  | 0.0183        | -               |
| 1.8956 | 9800  | 0.0189        | -               |
| 1.9052 | 9850  | 0.0164        | -               |
| 1.9149 | 9900  | 0.0171        | -               |
| 1.9246 | 9950  | 0.0172        | -               |
| 1.9342 | 10000 | 0.0199        | -               |
| 1.9439 | 10050 | 0.0167        | -               |
| 1.9536 | 10100 | 0.0184        | -               |
| 1.9632 | 10150 | 0.0181        | -               |
| 1.9729 | 10200 | 0.0176        | -               |
| 1.9826 | 10250 | 0.0161        | -               |
| 1.9923 | 10300 | 0.0176        | -               |
| 2.0019 | 10350 | 0.0162        | -               |
| 2.0116 | 10400 | 0.0191        | -               |
| 2.0213 | 10450 | 0.0166        | -               |
| 2.0309 | 10500 | 0.018         | -               |
| 2.0406 | 10550 | 0.018         | -               |
| 2.0503 | 10600 | 0.0206        | -               |
| 2.0600 | 10650 | 0.0187        | -               |
| 2.0696 | 10700 | 0.0191        | -               |
| 2.0793 | 10750 | 0.0165        | -               |
| 2.0890 | 10800 | 0.0181        | -               |
| 2.0986 | 10850 | 0.0169        | -               |
| 2.1083 | 10900 | 0.0202        | -               |
| 2.1180 | 10950 | 0.0176        | -               |
| 2.1277 | 11000 | 0.0205        | -               |
| 2.1373 | 11050 | 0.0174        | -               |
| 2.1470 | 11100 | 0.019         | -               |
| 2.1567 | 11150 | 0.0202        | -               |
| 2.1663 | 11200 | 0.019         | -               |
| 2.1760 | 11250 | 0.0196        | -               |
| 2.1857 | 11300 | 0.0191        | -               |
| 2.1954 | 11350 | 0.0204        | -               |
| 2.2050 | 11400 | 0.0166        | -               |
| 2.2147 | 11450 | 0.0197        | -               |
| 2.2244 | 11500 | 0.0161        | -               |
| 2.2340 | 11550 | 0.0204        | -               |
| 2.2437 | 11600 | 0.0166        | -               |
| 2.2534 | 11650 | 0.0172        | -               |
| 2.2631 | 11700 | 0.0183        | -               |
| 2.2727 | 11750 | 0.0194        | -               |
| 2.2824 | 11800 | 0.0185        | -               |
| 2.2921 | 11850 | 0.0157        | -               |
| 2.3017 | 11900 | 0.0172        | -               |
| 2.3114 | 11950 | 0.0167        | -               |
| 2.3211 | 12000 | 0.0184        | -               |
| 2.3308 | 12050 | 0.015         | -               |
| 2.3404 | 12100 | 0.0174        | -               |
| 2.3501 | 12150 | 0.0182        | -               |
| 2.3598 | 12200 | 0.017         | -               |
| 2.3694 | 12250 | 0.0175        | -               |
| 2.3791 | 12300 | 0.0177        | -               |
| 2.3888 | 12350 | 0.0179        | -               |
| 2.3985 | 12400 | 0.0169        | -               |
| 2.4081 | 12450 | 0.0169        | -               |
| 2.4178 | 12500 | 0.0176        | -               |
| 2.4275 | 12550 | 0.0195        | -               |
| 2.4371 | 12600 | 0.0168        | -               |
| 2.4468 | 12650 | 0.0178        | -               |
| 2.4565 | 12700 | 0.0157        | -               |
| 2.4662 | 12750 | 0.019         | -               |
| 2.4758 | 12800 | 0.0181        | -               |
| 2.4855 | 12850 | 0.0182        | -               |
| 2.4952 | 12900 | 0.0172        | -               |
| 2.5048 | 12950 | 0.0173        | -               |
| 2.5145 | 13000 | 0.018         | -               |
| 2.5242 | 13050 | 0.0181        | -               |
| 2.5338 | 13100 | 0.0186        | -               |
| 2.5435 | 13150 | 0.0195        | -               |
| 2.5532 | 13200 | 0.0194        | -               |
| 2.5629 | 13250 | 0.0158        | -               |
| 2.5725 | 13300 | 0.0157        | -               |
| 2.5822 | 13350 | 0.0155        | -               |
| 2.5919 | 13400 | 0.0179        | -               |
| 2.6015 | 13450 | 0.0178        | -               |
| 2.6112 | 13500 | 0.0181        | -               |
| 2.6209 | 13550 | 0.0161        | -               |
| 2.6306 | 13600 | 0.0181        | -               |
| 2.6402 | 13650 | 0.021         | -               |
| 2.6499 | 13700 | 0.0182        | -               |
| 2.6596 | 13750 | 0.0171        | -               |
| 2.6692 | 13800 | 0.0171        | -               |
| 2.6789 | 13850 | 0.017         | -               |
| 2.6886 | 13900 | 0.0174        | -               |
| 2.6983 | 13950 | 0.0175        | -               |
| 2.7079 | 14000 | 0.0179        | -               |
| 2.7176 | 14050 | 0.0174        | -               |
| 2.7273 | 14100 | 0.0185        | -               |
| 2.7369 | 14150 | 0.0192        | -               |
| 2.7466 | 14200 | 0.0172        | -               |
| 2.7563 | 14250 | 0.0195        | -               |
| 2.7660 | 14300 | 0.0162        | -               |
| 2.7756 | 14350 | 0.0215        | -               |
| 2.7853 | 14400 | 0.0188        | -               |
| 2.7950 | 14450 | 0.0177        | -               |
| 2.8046 | 14500 | 0.0169        | -               |
| 2.8143 | 14550 | 0.0161        | -               |
| 2.8240 | 14600 | 0.0185        | -               |
| 2.8337 | 14650 | 0.0179        | -               |
| 2.8433 | 14700 | 0.0183        | -               |
| 2.8530 | 14750 | 0.0183        | -               |
| 2.8627 | 14800 | 0.0188        | -               |
| 2.8723 | 14850 | 0.017         | -               |
| 2.8820 | 14900 | 0.0169        | -               |
| 2.8917 | 14950 | 0.0179        | -               |
| 2.9014 | 15000 | 0.0174        | -               |
| 2.9110 | 15050 | 0.0191        | -               |
| 2.9207 | 15100 | 0.0169        | -               |
| 2.9304 | 15150 | 0.0185        | -               |
| 2.9400 | 15200 | 0.0178        | -               |
| 2.9497 | 15250 | 0.0198        | -               |
| 2.9594 | 15300 | 0.0178        | -               |
| 2.9691 | 15350 | 0.0201        | -               |
| 2.9787 | 15400 | 0.0172        | -               |
| 2.9884 | 15450 | 0.0168        | -               |
| 2.9981 | 15500 | 0.0182        | -               |

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