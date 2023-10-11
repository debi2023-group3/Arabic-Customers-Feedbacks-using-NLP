import torch
import pandas as pd
import os
import json
from experiments.classifier_exp import Classification
from sklearn.model_selection import train_test_split
import logging

SEED = 2023
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

with open('./classifier_params.json', 'r') as config_file:
    args = json.load(config_file)
    logger.debug("File Parameters loaded successfully")

exp = Classification(args)
train_data = pd.read_csv(args['data_args']['train_data_path'])
test_data = pd.read_csv(args['data_args']['test_data_path'])

train_data, eval_data = train_test_split(train_data, test_size=0.1, shuffle=True)

history = exp.train(train_data)
torch.cuda.empty_cache()
model_performance = exp.evaluate(eval_data)
testing_performance = exp.predict(test_data)
# exp.retrain(train_data)