import pubchempy as pcp
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import csv

class ChemBERTaEmbedding(nn.Module):
    def __init__(self, local_model_path='ChemBERTa-77M-MLM', embedding_dim=384):
        super(ChemBERTaEmbedding, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(local_model_path)
        self.model = RobertaModel.from_pretrained(local_model_path)
        self.fc = nn.Linear(embedding_dim, 4)  # 全连接层，将向量嵌入到一个数字

    def forward(self, smiles):
        inputs = self.tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state.mean(dim=1).squeeze()
        embedding = self.fc(pooled_output)
        return embedding

def find_smiles(drug_name):
    compound = pcp.get_compounds(drug_name, "name")  # 使用药物名称查询
    if compound:
        return compound[0].canonical_smiles
    else:
        return -1

# 对smiles进行数字编码
def smiles_encoder(smiles, model_path='ChemBERTa-77M-MLM'):
    if smiles == -1:
        return 0
    else:
        model = ChemBERTaEmbedding(model_path)
        return model(smiles)

# 首先安装必要库（在终端中运行）
# conda install -c conda-forge rdkit 或 pip install rdkit-pypi

from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import Image  # 用于Jupyter直接显示

def smiles_to_image(smiles, 
                   image_size=(300, 300),
                   filename='molecule.png',
                   show_atom_numbers=False,
                   legend=''):
    try:
        # 生成分子对象
        mol = Chem.MolFromSmiles(smiles)
        
        # 可选：添加氢原子显示
        mol = Chem.AddHs(mol)
        
        # 绘制参数设置
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*image_size)  # 使用Cairo渲染
        opts = drawer.drawOptions()
        
        # 自定义显示样式
        opts.atomLabelFontSize = 18
        opts.bondLineWidth = 2
        opts.useBWAtomPalette()  # 黑白显示
        
        if show_atom_numbers:
            for atom in mol.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))
        
        # 绘制分子
        drawer.DrawMolecule(mol, legend=legend)
        drawer.FinishDrawing()
        
        # 保存图像
        with open(filename, 'wb') as f:
            f.write(drawer.GetDrawingText())
            
        return Image(filename=filename)
    
    except Exception as e:
        print(f"可视化失败: {str(e)}")
        return None


if __name__ == '__main__':
    import pandas as pd
    df=pd.read_csv('meta_fin_1.csv',header=0)
    csv_file = 'drug_embedding.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['drug_name', 'smiles', 'embedding'])  # 写入表头

        for drug_name in drug_names:
            smiles = find_smiles(drug_name)
            embedding = smiles_encoder(smiles)
            if smiles != 0:
                writer.writerow([drug_name, smiles, embedding.item()**2])  # 写入每行数据
            else:
                writer.writerow([drug_name, smiles, 0])

    print(f"Data has been saved to {csv_file}")