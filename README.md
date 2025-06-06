# MetaAMR-Metric

MetaAMR-Metric is a cutting-edge metric-based meta-learning framework specifically designed for predicting antimicrobial resistance (AMR). It intelligently matches the most suitable machine learning model to the characteristics of each antibiotic, significantly enhancing the AMR prediction process.

## Environment Requirements

```text
python >= 3.8  
torch >= 2.0.0  
scikit-learn >= 1.3.2
torch-geometric>=2.0.0
rdkit
numpy
pandas
biopython
```

## Project Structure

```text
MetaAMR-Metric/
├── Meta/                         # Meta-learning module and model training code  
│   ├── main.py                  # Main training script for the meta-learning model  
│   ├── Drugencoder.py           # Drug molecular structure encoder  
│   ├── GAN.py                   # GAN-based data augmentation module  
│   ├── PrototypeNet.py          # Prototype network model definition  
│
├── Models/                       # Basic machine learning models  
│   ├── GNN/                     # Implementation of the GNN model  
│   ├── ML_models.py             # Implements Logistic Regression, Random Forest, and SVM  
│
├── README.md                    # Project description and documentation  
```
