# MetaAMR-Metric

MetaAMR-Metric is a cutting-edge metric-based meta-learning framework designed specifically for predicting antimicrobial resistance (AMR). It intelligently matches optimal machine learning models to the characteristics of different antibiotics, revolutionizing the AMR prediction process.

## Environment requirement
   
    python>=3.8
    torch>=2.0.0
    sklearn>=1.3.2
    

## Documents

  MetaAMR-Metric/
    ├── Meta/     # 包含评估模型的实现  
       ├── main.py/     # 元学习模型训练代码  
       ├── Drugencoder.py/     # 药物分子结构编码  
       ├── GAN.py/     # 数据增强模型 
       ├── PrototypeNet.py/     # 原型网络模型 
    ├── Models/     # 四种模型 
       ├── GNN/     # GNN模型的实现
          ├── GNN/     # GNN模型的实现
          ├── GNN/     # GNN模型的实现
          ├── GNN/     # GNN模型的实现
       ├── ML_models.py/     # LR,RF,SVM三种模型的实现  
    ├── README.md

