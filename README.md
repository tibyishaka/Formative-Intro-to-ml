# Formative One  Introduction to Machine Learning
**IBYISHAKA TRESOR Thierry Alain**

A complete ML classification pipeline applied to the [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) (UCI). The notebook covers classical machine learning models and a neural network built from scratch in NumPy, with full evaluation and comparative analysis.

---

## Notebook Structure

| Part | Content |
|------|---------|
| 1 | Dataset selection and justification |
| 2 | Classical ML models  Logistic Regression & Random Forest |
| 3 | 3-layer neural network implemented from scratch in NumPy |
| 4 | Comparative analysis, discussion, and references |

---

## Dataset

**Banknote Authentication**  1,372 samples, 4 continuous features extracted from wavelet-transformed images of banknotes:

| Feature | Description |
|---------|-------------|
| `variance` | Variance of wavelet-transformed image |
| `skewness` | Skewness of wavelet-transformed image |
| `curtosis` | Kurtosis of wavelet-transformed image |
| `entropy` | Entropy of the image |
| `class` | 0 = Genuine, 1 = Forged |

---

## Models

### Classical ML (scikit-learn)
| Model | Hyperparameter Variation |
|-------|--------------------------|
| Logistic Regression | `C=1.0` (default) vs `C=0.01` (strong L2 regularization) |
| Random Forest | `n=10, depth=3` (shallow) vs `n=100, depth=None` (full) |

### Neural Network from Scratch (NumPy only)
- **Architecture**: Input(4) → Hidden1(ReLU, 16) → Hidden2(ReLU, 8) → Output(Sigmoid, 1)
- **Loss**: Binary Cross-Entropy with numerical stability clipping
- **Initialization**: He (Kaiming) initialization
- **Optimizer**: Vanilla Batch Gradient Descent
- **Experiments**: 3 variations across learning rate and architecture width
- **Validation**: Per-epoch training vs validation loss/accuracy tracked

---

## Evaluation

All models evaluated on a held-out 20% test set with:
- Accuracy, Precision, Recall, F1-score (comparison table)
- Confusion matrices for all 5 model configurations
- ROC / AUC curves for all 5 model configurations
- Training vs validation learning curves (NN experiments)

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

The dataset is loaded automatically from the UCI repository URL  no manual download needed.

---

## Running the Notebook

```bash
jupyter notebook Assignment1_IBYISHAKA-Alain.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/)  all cells run without errors.

---

## References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *IEEE ICCV*, 1026–1034.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
6. Howard, J. & Gugger, S. (2020). *Practical Deep Learning for Coders*. fast.ai. https://course.fast.ai/
