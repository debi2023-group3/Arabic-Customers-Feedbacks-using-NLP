import torch 
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class CustomerDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer_name, max_len):
        super().__init__()
        
        self.texts = texts
        self.labels = labels
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, indx):
        # encoded_id, atten_mask 
        encoded  = self.tokenizer(
                                text= self.texts[indx], 
                                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                max_length=self.max_len,        # Max length to truncate/pad
                                padding = 'max_length',         # Pad sentence to max length
                                truncation=True,
                                return_tensors='pt',            # Return PyTorch tensor
                                return_attention_mask=True,     # Return attention mask
                                )

        # Create a dictionary to hold the data
        data_dict = {
            "text": self.texts[indx],
            "encoded_id": encoded['input_ids'].flatten(),
            "atten_mask": encoded['attention_mask'].flatten(),
        }

        # Check if labels are available
        if self.labels is not None:
            data_dict["label"] = self.labels[indx]

        return data_dict
        
        
        
        
        
        
        
        
        
        
    