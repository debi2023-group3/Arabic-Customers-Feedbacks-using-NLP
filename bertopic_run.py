import torch
import json
import logging
import pandas as pd
from experiments.topic_modeling_exp import TopicModeling
from sklearn.model_selection import train_test_split

SEED = 2023
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('./bertopic_params.json', 'r') as config_file:
    args = json.load(config_file)
    logger.debug("File Parameters loaded successfully")

labeled_data = pd.read_csv("data/raw/labeled_data.csv")    
unlabeled_data = pd.read_csv("data/raw/unlabeled_data.csv")    

exp = TopicModeling(args)

train_data, val_data = train_test_split(labeled_data, test_size=0.2, shuffle=True)
# train
exp.fit(train_data)
torch.cuda.empty_cache()
# validation
exp.validation(val_data)
# # test
# exp.transform(unlabeled_data)