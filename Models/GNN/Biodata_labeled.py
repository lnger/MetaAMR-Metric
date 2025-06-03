import numpy as np
from Bio import SeqIO
import Train
import torch
import torch_geometric.transforms as T
import torch_geometric.utils as ut
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from utils import *
import sys
import os

# 替换为你的模块实际路径
module_path = 'build/lib.linux-x86_64-cpython-311/encode_seq.cpython-311-x86_64-linux-gnu.so'

if module_path not in sys.path:
    sys.path.append(module_path)

import encode_seq


class BipartiteData(Data):
    def _add_other_feature(self, other_feature) :
        self.other_feature = other_feature

    def __inc__(self, key, value, store):
        if key == 'edge_index':
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)


class GraphDataset():
    
    def __init__(self, pnode_feature, fnode_feature, other_feature, edge,graph_label):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label


    def process(self):
        data_list = []
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)
            
            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)
            
            data_list.append(data)

        return data_list


class GraphDatasetInMem(InMemoryDataset):

    def __init__(self, pnode_feature, fnode_feature, other_feature, edge,graph_label, root, transform=None, pre_transform=None):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label
        super(GraphDatasetInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['test.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)

            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)

            data_list.append(data)
        data, slices = self.collate(data_list)  # Here used to be [data] for one graph
        torch.save((data, slices), self.processed_paths[0])

        return data_list


class Biodata:
    def __init__(self,sequences,balance,drug_id, K=3, d=3,label_feature_file='../Data/data.csv', seqtype="Mutation"):
        self.dna_seq = {}
        label_file=pd.read_csv(label_feature_file,header=0)
        label_file=label_file[['File name','AMP Concl', 'AUG Concl', 'AXO Concl', 'CHL Concl',
                               'CIP Concl', 'COT Concl', 'FIS Concl', 'FOX Concl',
                               'NAL Concl', 'STR Concl', 'TET Concl']]
        labels_dict = label_file.set_index('File name').T.to_dict(orient='list')
        self.labels_dict = {k: v[drug_id+1] for k, v in labels_dict.items()}
        self.labels=[]
        
        for seq_id, seq in sequences.items():
            self.dna_seq[seq_id] =seq
            self.labels.append(labels_dict[seq_id+'.fasta'])
        
        self.balance_samples(ratio=balance)
        
        self.other_feature = None

        self.K = K
        self.d = d
        # self.drugtype = drugtype
        self.seqtype = seqtype
        self.base=4
        if self.seqtype=='Mutation':
            self.base=16
        elif self.seqtype=='ATCGN':
            self.base=5

        self.labels=np.array(self.labels)
        print(self.labels)
    
    def encode(self, thread=10, save_dataset=True, save_path="./"):
        dataset = []
        print("Encoding sequences...")
        seq_list = list(self.dna_seq.values())
        seq_ids = list(self.dna_seq.keys())

        from tqdm import tqdm
        progress_bar = tqdm(total=len(seq_list), desc="Processing sequences", unit="seq")
        with ThreadPoolExecutor(max_workers=thread) as executor:
            futures = [
                executor.submit(
                    encode_seq.create_graph_data,
                    seq, 
                    K=self.K,
                    d=self.d,
                    seqtype=self.seqtype
                ) 
                for seq in seq_list
            ]
            x_list, edge_idx_list, edge_attr_list = [], [], []
            for future in futures:
                x, edge_idx, edge_attr = future.result()
                x_list.append(x)
                edge_idx_list.append(edge_idx)
                edge_attr_list.append(edge_attr)
                progress_bar.update(1)
        progress_bar.close()
        
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(x_list)):
            data = Data(
                x=torch.FloatTensor(x_list[i]),
                edge_index=torch.LongTensor(edge_idx_list[i]),
                edge_attr=torch.FloatTensor(edge_attr_list[i]),
                y=torch.tensor([self.labels[i]])
            )
            if save_dataset:
                torch.save(data, os.path.join(save_path, f"{seq_ids[i]}.pt"))
            
            dataset.append(data)
        
        return dataset
    
    def balance_samples(self, ratio=1, positive_label=1):
        from collections import defaultdict
        import random
        label_groups = defaultdict(list)
        for seq_id in self.dna_seq.keys():
            label = self.labels_dict.get(seq_id + '.fasta')
            if label is not None:
                label_groups[label].append(seq_id)

        negative_label = 1 - positive_label
        
        pos_count = len(label_groups.get(positive_label, []))
        neg_count = len(label_groups.get(negative_label, []))
        try:
            pos_num = min(pos_count, neg_count // ratio)
            neg_num = pos_num * ratio
        except KeyError:
            raise ValueError("Invalid label value")
        sampled_seqs = []
        sampled_seqs.extend(random.sample(label_groups[positive_label], pos_num))
        sampled_seqs.extend(random.sample(label_groups[negative_label], neg_num))
        random.shuffle(sampled_seqs)
        self.dna_seq = {seq_id: self.dna_seq[seq_id] for seq_id in sampled_seqs}
        self.labels = [self.labels_dict[seq_id+'.fasta'] for seq_id in sampled_seqs]
        
        print(f"Balanced dataset (1:{ratio}): {pos_num + neg_num} samples "
            f"({positive_label}: {pos_num}, {negative_label}: {neg_num})")


if __name__ == '__main__':
    core_path='core.aln'

    with open(core_path, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        
    ref_record = records[0]
    ref_id = ref_record.id
    ref_seq = str(ref_record.seq)
    sequences={}

    sequences.update({record.id: record.seq for record in records[1:]})

    K=3
    d=5

    drugs= ['AMP', 'AUG', 'AXO', 'CHL', 'CIP', 'COT', 'FIS', 'FOX', 'NAL', 'STR', 'TET']
    balances=[1,2,3,4]
    for i, drug in enumerate(drugs):
        for balance in balances:
            print("\n=== drug:{}  balance:{} ===".format(drug,balance))
            data = Biodata(sequences,balance,i,K=K,d=d,seqtype="ATCGN")
            dataset = data.encode(thread=10,save_dataset=False)

            device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

            model = Train.CustomModel(input_dim=K*5+1, hidden_dim_1=64, hidden_dim_2=128, output_dim=2, edge_dim=d).to(device)
            model=Train.train(dataset, model,batch_size=128,epoch_n=500, weighted_sampling=True,device=device)

