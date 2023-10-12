import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from models.bertopic import BERTopicModel
from utils.evaluation import evaluate_classifier
from data_factory.text_preprocessing import TextPreprocessor

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class TopicModeling(object):
    
    def __init__(self = None, args = None):
        super(TopicModeling, self).__init__()
        
        self.model_config = args['model_config']
        self.data_args = args['data_args']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probs = None
        self.topics = None
        # Store Preprocessing steps in list to be tracked by MLFlow
        self.prep_args = []
        for key, value in self.data_args.items():
            if value: 
                self.prep_args.append(key)
        self.prep_args = self.prep_args[:-9]

        with open('./class_mapping.json') as json_file:
                self.reverse_classes = json.load(json_file)

        self.classes = {v: int(k) for k, v in self.reverse_classes.items()}
# ================================================================================= #
    def setup(self):
        self.client = MlflowClient()

        experiment_name = "Arabic Clustering Using BERTopic"
        tags = {"language":"Arabic", "technique":"NLP", "model":"BERTopic", "task":"text Clustering"}

        # Retraive experiment by its name
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Uncomment to create experiment for the first time
            self.experiment_id = self.client.create_experiment(
                                    experiment_name,
                                    artifact_location=Path.cwd().joinpath("mlruns").as_uri())
        else:
            self.experiment_id = experiment.experiment_id
        
        for key, value in tags.items():
            self.client.set_experiment_tag(self.experiment_id, key, value)

        # Fetch experiment metadata information
        experiment = self.client.get_experiment(self.experiment_id)
        print()
        print('='*50)
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(self.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
        print('='*50)
        print()
# ================================================================================= #
    def get_cleaned_data(self, data):
        
        preprocessor = TextPreprocessor(data, self.data_args)
        self.processed_data = preprocessor.preprocess()
        reviews = list(self.processed_data[self.data_args['text_col']].values)

        # label encoder for label if exist
        if self.model_config['task'] in ['supervised', 'cls']:

            self.processed_data[self.data_args['target_col']] = self.processed_data[self.data_args['target_col']].map(self.classes)
            labels = list(self.processed_data[self.data_args['target_col']].values)
            
        else: 
            labels = None
        
        return reviews, labels
# ================================================================================= #
    def fit(self, data):
        
        # 1. load cleaned reviews 
        self.reviews, self.labels = self.get_cleaned_data(data)

        # 2. Define Model
        self.model = BERTopicModel(self.model_config)
        # pass model to GPU
        if self.data_args['use_gpu']:
            print('Use GPU:{}'.format(self.device))
            self.model = self.model.to(self.device)
            
        # start tracking the experiment using MLflow
        self.setup()
        mlflow.start_run(experiment_id = self.experiment_id)
        # self.run_id = mlflow.active_run().info.run_id
        mlflow.pytorch.autolog()
        # store hyperparameters in MLflow
        mlflow.log_params(self.model_config)
        mlflow.log_param("Preprocessing_steps", self.prep_args)
        mlflow.log_dict(self.reverse_classes, "classes_mapping.json")

        # 3. Make prediction 
        print(f">> Start training the model ...")
        self.topics, self.probs = self.model(self.reviews, self.labels)
        self.processed_data['Topic'] = self.topics

        # 4. Evaluate if (cls):
        if self.model_config['task'] in ['supervised', 'cls']:
            performance = evaluate_classifier(self.labels, self.topics, self.classes)
            del performance['confusion_matrix']
            del performance['classification_report']
            mlflow.log_metrics(performance)
        # Log dataset in MLflow 
        raw_dataset: PandasDataset = mlflow.data.from_pandas(data)
        mlflow.log_input(raw_dataset, context="Raw data Before Preprocessing")
        annot_dataset: PandasDataset = mlflow.data.from_pandas(self.processed_data)
        mlflow.log_input(annot_dataset, context="Topic Modeling Training Dataset with annotation")
        
        if self.data_args['save_annotation'] is not None:

            print(f'''>> Saving dataset with annotated topics in `{self.data_args['save_annotation']}` directory.... ''')
            os.makedirs(self.data_args['save_annotation'], exist_ok=True)
            self.processed_data.to_csv(f"{self.data_args['save_annotation']}/train_labeled_data.csv", index=False)

        if self.data_args['checkpoints_dir'] is not None:

            print(f'''>> Saving trained model in `{self.data_args['checkpoints_dir']}` directory.... ''')
            os.makedirs(self.data_args['checkpoints_dir'], exist_ok=True)
            model_path = self.model.save(self.data_args['checkpoints_dir'])
            signature = infer_signature(np.array(self.reviews), np.array(self.topics))
            mlflow.pytorch.log_model(self.model, 'BERTopic model', registered_model_name= model_path.split('/')[-1], signature=signature)
            
        if self.data_args['results_dir'] is not None:

            print(f'''>> Saving topic results in `{self.data_args['results_dir']}` directory.... ''')
            os.makedirs(self.data_args['results_dir'], exist_ok=True)
            file_path = self.model.save_results(self.data_args['results_dir'])
            mlflow.log_artifact(file_path,  "Training Results")

        if self.data_args['visual_dir'] is not None:

            print(f'''>> Saving figures in `{self.data_args['visual_dir']}` directory.... ''')
            docs = self.processed_data[self.data_args['text_col']].values
            os.makedirs(self.data_args['visual_dir'], exist_ok=True)
            try:
                classes = [self.reverse_classes.get(str(label)) for label in self.labels]
            except TypeError:
                classes=None
            self.model.visualize_topics(docs, self.data_args['visual_dir'], self.topics, classes)
            
        # mlflow.end_run()
        
        return self.processed_data
# ================================================================================= #
    def validation(self, data):
        
        reviews, labels = self.get_cleaned_data(data)
        print(f">> Start validating the model ...")
        topics, embed_documents = self.model.transform(reviews)
        
        result = pd.DataFrame()
        result['Review'] = reviews
        result['Label'] = labels
        result['Topic'] = topics
        
        mlflow.start_run(experiment_id = self.experiment_id, nested=True)
        if self.model_config['task'] in ['supervised', 'cls']:
            performance = evaluate_classifier(labels, topics, self.classes)
            del performance['confusion_matrix']
            del performance['classification_report']
            mlflow.log_metrics(performance)

        # if self.data_args['visual_dir'] is not None:
        #     os.makedirs(f"{self.data_args['visual_dir']}/validation", exist_ok=True)
        #     classes = [self.reverse_classes.get(str(label)) for label in self.labels]
        #     self.model.visualize_topics(reviews, self.data_args['visual_dir'], self.topics, classes, embed_documents, 'val')
        
        if self.data_args['save_annotation'] is not None:
            os.makedirs(self.data_args['save_annotation'], exist_ok=True)
            result.to_csv(f"{self.data_args['save_annotation']}/val_labeled_data.csv", index=False)
            
            annot_dataset: PandasDataset = mlflow.data.from_pandas(self.processed_data)
            mlflow.log_input(annot_dataset, context="Topic Modeling Validation Dataset with annotation")

        mlflow.end_run()

        return result
# ================================================================================= #
    def transform(self, data):
        
        preprocessor = TextPreprocessor(data, self.data_args)
        processed_data = preprocessor.preprocess()
        reviews = list(processed_data[self.data_args['text_col']].values)

        topics, _ = self.model.transform(reviews)

        result = pd.DataFrame()
        result['Review'] = reviews
        result['Topic'] = topics

        if self.data_args['save_annotation'] is not None:
            os.makedirs(self.data_args['save_annotation'], exist_ok=True)
            result.to_csv(f"{self.data_args['save_annotation']}/test_labeled_data.csv", index=False)
        
        return result

    