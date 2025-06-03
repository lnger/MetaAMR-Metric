# MetaAMR-Metric

MetaAMR-Metric is a cutting-edge metric-based meta-learning framework designed specifically for predicting antimicrobial resistance (AMR). It intelligently matches optimal machine learning models to the characteristics of different antibiotics, revolutionizing the AMR prediction process.

## Environment requirement
   
    python>=3.8
    torch>=2.0.0
    sklearn>=1.3.2
    

## Documents
  MetaAMR-Metric/
    ├── Meta/      # 存放元数据和配置文件
       ├── main.py/
       ├── Drugencoder.py/
       ├── GAN.py/
       ├── PrototypeNet.py/
    ├── Models/     # 包含评估模型的实现   
       ├── GNN/
          ├── GNN/
       ├── ML_models.py/
    ├── README.md            # 项目说明文档
    ├── requirements.txt     # Python 依赖列表
    └── setup.py             # 安装脚本（如适用）
