import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BERTClassifier(nn.Module):    
    def __init__(self, args = None):
        super(BERTClassifier, self).__init__()
        
        self.bert_name = args["bert_name"]
        self.n_layers = args["n_layers"]
        self.n_nodes = args["n_nodes"]
        self.n_classes = args["n_classes"]
        
        self.bert = AutoModel.from_pretrained(self.bert_name)
        self.dropout = nn.Dropout(p=0.2)
        self.dens1 = nn.Linear(self.bert.config.hidden_size, self.n_nodes)
        self.dens2 = nn.Linear(self.n_nodes, self.n_nodes)
        self.relu  = nn.ReLU()
        self.output_layer = nn.Linear(self.n_nodes, self.n_classes)
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, ids, atten):
        
        output = self.bert(ids, atten)
        # Extract the last hidden state of the token `[CLS]` for classification task
        output = self.dens1(output.last_hidden_state[:, 0, :])  
        self.dropout(output)
        
        for i in range(self.n_layers):
            output = self.relu(output)
            output = self.dens2(output)
            self.dropout(output)
            
        return self.output_layer(output)
            
        
        
        