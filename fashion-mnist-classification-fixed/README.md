# 👕  Fashion-MNIST Case Study

Apparel classification using a Convolutional Neural Network (CNN) on the **Fashion-MNIST dataset**.  
This project achieves **~93% test accuracy** and also provides classical ML baselines for comparison.

---

## ⚙️ Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train CNN
```bash
python Main.py --train --epochs 15 --batch 128
```

### Run inference grid (visualize predictions)
```bash
python Main.py --infer --samples 9
```

### Run classical ML baselines
```bash
python Main.py --baselines
```

---

## 🏆 Results / Outputs
- **Test Accuracy:** ~93%  
- Artifacts are saved in the `artifacts/` folder:
  - `fashion_cnn.h5` (best model)
  - `fashion_cnn_final.h5` (final model)
  - `training_curves.png`
  - `confusion_matrix.png`
  - `misclassifications.png`
  - `classification_report.txt`
  - `summary.txt`
  - `label_map.txt`
  - `inference_grid.png`

**Training Curves**  
![Training Curves](artifacts/training_curves.png)

**Confusion Matrix**  
![Confusion Matrix](artifacts/confusion_matrix.png)

**Sample Misclassifications**  
![Misclassifications](artifacts/misclassifications.png)
