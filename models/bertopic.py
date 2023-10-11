import os
import torch
import torch.nn as nn
import umap
import mlflow
from datetime import datetime
from scipy.cluster import hierarchy as sch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from utils.evaluation import evaluate_classifier

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import ZeroShotClassification
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from hdbscan import HDBSCAN
import arabicstopwords.arabicstopwords as stp
import plotly.io as pio
from plotly import graph_objects as go

SEED = 2023

class BERTopicModel(nn.Module):
    
    def __init__(self, configs):
        super(BERTopicModel, self).__init__()
        self.configs = configs  
        self.corpus_embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, docs, labels=None):
        self.model = BERTopic( 
            language='use_multilingual',
            n_gram_range=self.configs['ngram_range'], 
            min_topic_size  = self.configs['min_topic_size'],
            nr_topics = self.configs['n_topics'],
            # the number of words per topic that you want to be extracted.
            top_n_words = 15, 
            calculate_probabilities  = True,
            # Step 2 - Reduce dimensionality
            umap_model=self._Dimensionality_reduction(),         
            # Step 3 - Cluster reduced embeddings
            hdbscan_model=self._Clustering(),         
            # Step 4 - Tokenize topics
            vectorizer_model=self._Vectorization(),      
            # Step 5 - Extract topic words
            ctfidf_model= ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True),   
            # Step 6 - Fine-tune topics
            representation_model=self._Representation())  
        
        self.corpus_embeddings = self._Embedding(docs)
        topics, probs = self.model.fit_transform(docs, self.corpus_embeddings, y=labels)
        
        if self.configs['task'] == 'guided':
            self.model.update_topics(docs, 
                        seed_topic_list=[
                            ["ممكن", "ياريت", "لكن","اتمنى", "تعديل", "ينقص","خدمة","ارجو","تفعيل","اضافة", "يحتاج", "تنظيم"], # suggession
                            ["جميل", "ممتاز","سهل", "رائع", "جيد", "محترم", "مفيد", "كويس","شكرا"], # positive
                            ["زفت", "سيئ", "خرا", "بطئ","حرامي", "نصب","سئ", "زباله", "فشل"],# negative
                            ["شحن", "حساب", "رصيد", "فواتير", "خدمة", "فيزا", "مشكلة", "سحب", "رفض", "تكلفه", "خصم"], #issue
                            ["لماذا", "ماهي", "ليه", "هل", "كيفية", "عايز", "كيف", "اريد", "ازاي", "استفسار", "?"],] )#Quesion)

        return topics, probs
    
    def transform(self, docs, labels=None):
        embed_documents = self._Embedding(docs)
        if labels is None:
            topics, _ = self.model.transform(docs, embed_documents)
        else:
            topics, _ = self.model.transform(docs, embed_documents, y=labels)
        
        return topics, embed_documents#, y_mapped
    
    def save(self, checkpoints_dir):
        
        setting = f"{self.configs['embd_technique']}_embed_{self.configs['n_topics']}topics_{self.configs['n_components']}dim_{self.configs['clustering_model_name']}"
        full_path = os.path.join(checkpoints_dir, setting)
        self.model.save(full_path, serialization="pickle")
        #save(full_path , serialization="pytorch", save_ctfidf=True)
        return full_path
    
    def visualize_topics(self, docs, visual_dir, topics, labels=None, embed_documents=None, func='train'):

        fig = self.model.visualize_topics()
        # save graph locally
        fig.write_html(os.path.join(visual_dir, f"itertopics_distance_map.html"))
        # save on MLflow
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Itertopics distance map.html")

        if func == 'train':
            embed_documents = self.corpus_embeddings
        
        reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embed_documents)
        fig = self.model.visualize_documents(docs=docs, reduced_embeddings=reduced_embeddings)
        fig.write_html(os.path.join(visual_dir, f"documents_and_topics.html"))
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Documents and topics.html")

        fig = self.model.visualize_heatmap()
        fig.write_html(os.path.join(visual_dir, f"similarity_matrix.html"))
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Similarity matrix.html")

        fig = self.model.visualize_barchart(height=700)
        fig.write_html(os.path.join(visual_dir, f"topic_word_scores.html"))
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Topic word scores.html")

        fig = self.model.visualize_term_rank()
        fig.write_html(os.path.join(visual_dir, f"term_score_decline_per_topic.html"))
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Term score decline per topic.html")

        topic_distr, _ = self.model.approximate_distribution(docs, min_similarity=0)
        fig = self.model.visualize_distribution(topic_distr[0])
        fig.write_html(os.path.join(visual_dir, f"visualize_distribution.html"))
        fig = go.Figure(fig)
        mlflow.log_figure(fig, f"Visualize distribution.html")

        # linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
        # hierarchical_topics = self.model.hierarchical_topics(docs, linkage_function=linkage_function)
        # fig = self.model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        # fig.write_html(os.path.join(visual_dir, f"hierarchical_clustering.html"))
        # fig = go.Figure(fig)
        # mlflow.log_figure(fig, f"Hierarchical clustering.html")

        if self.configs['task'] in ['semi-supervised', 'supervised', 'cls']:
            topics_per_class=self.model.topics_per_class(docs, classes=labels)   
            fig = self.model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
            fig.write_html(os.path.join(visual_dir, f"topics_per_class.html"))
            fig = go.Figure(fig)
            mlflow.log_figure( fig, "Topic per class.html")

    def save_results(self, results_dir):
        file_path = os.path.join(results_dir, "BERTopic_results.txt")
        with open(file_path, 'w') as file:
            file.write(f"\n\n> Experiment Name: 'Topic Modeling using BERTopic' \n")
            file.write(f"> Date: {datetime.now()}\n")
            file.write(f"> Topic Info: \n")
            file.write(self.model.get_topic_freq().to_string())
            file.write('\n')
            file.write('='*150)
        return file_path
                
    def _Embedding(self, docs):
        if self.configs['embd_technique'] == 'sent':
            try:
                model_embedding = SentenceTransformer(self.configs['embed_model_name'])
                return model_embedding.encode(docs)
            
            except ModuleNotFoundError:
                print("Invalid model name: Visit the official Sentence Transformers (https://www.sbert.net/docs/pretrained_models.html)")
                return None  # Return None or raise an exception, depending on your use case
        
        elif self.configs['embd_technique'] == 'word':
            tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01",
                                                        do_lower_case=False,
                                                        do_basic_tokenize=True,)
            encoded_id = tokenizer(
                                docs,
                                padding=True,
                                truncation=True, 
                                max_length=self.configs['max_length'], 
                                return_tensors="pt")
            
            embed_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv01") 
            embed_model.to(self.device)
            with torch.no_grad():
                inputs = {key: val.to(self.device) for key, val in encoded_id.items()}
                outputs = embed_model(**inputs)
                embeddings = outputs.last_hidden_state  
            embedded_data = embeddings.view(len(docs), -1)
            embedded_data = embedded_data.cpu().numpy()
            return embedded_data

    def _Dimensionality_reduction(self):

        if self.configs['dim_model_name'] == 'pca':
            return PCA(n_components=self.configs['n_components'], random_state=SEED)
        
        elif self.configs['dim_model_name'] == 'umap':
            return umap.UMAP(n_neighbors=15, n_components=self.configs['n_components'], 
                            metric='cosine', low_memory=False, random_state=SEED)
        elif self.configs['task'] == 'cls':
            return BaseDimensionalityReduction()
        
    def _Clustering(self):
        """
        Initialize and return a clustering model based on the specified technique.

        Args:
            model_name (str): Model name for clustering.
            max_iter (int): Maximum number of iterations (used for KMeans).

        Returns:
            KMeans or HDBSCAN or AgglomerativeClustering or SpectralClustering: Initialized clustering model.
        """
        if self.configs['clustering_model_name'] == 'kmeans':
            return KMeans(n_init='auto', init='k-means++', max_iter=10, 
                            n_clusters=self.configs['n_topics'], random_state=SEED)
        
        elif self.configs['clustering_model_name'] == 'hdbscan':
            return HDBSCAN(min_cluster_size=self.configs['min_topic_size'], 
                            metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        
        elif self.configs['clustering_model_name'] == 'agglomerative':
            return AgglomerativeClustering(n_clusters=self.configs['n_topics'])
        
        elif self.configs['clustering_model_name'] == 'spectral':
            return SpectralClustering(n_clusters=self.configs['n_topics'])
        
        elif self.configs['clustering_model_name'] == 'lr':
            return  LogisticRegression()
        
        elif self.configs['clustering_model_name'] == 'nb':
            return  MultinomialNB()

    def _Vectorization(self):
        """
        Initialize and return a vectorization model based on the specified n-gram range.

        Args:
            ngram_range (tuple): Range for n-grams.

        Returns:
            CountVectorizer: Initialized CountVectorizer model.
        """
        vectorizer = CountVectorizer(stop_words=self._Stopwords())
        return vectorizer
    
    def _Stopwords(self):

        arabic_stopwords = stp.stopwords_list()
        arabic_stopwords.remove('لا')
        arabic_stopwords.remove('لكن')
        arabic_stopwords.append('تطبيق')
        arabic_stopwords.append('التطبيق')
        arabic_stopwords.append('برنامج')
        arabic_stopwords.append('البرنامج')
        arabic_stopwords.append('ابلكيشن')
        arabic_stopwords.append('الابلكيشن')
        arabic_stopwords.append('جد')
        arabic_stopwords.append('جدا')
        arabic_stopwords.append('تطبيقات')
        arabic_stopwords.append('برامج')
        arabic_stopwords.append('الي')
        arabic_stopwords.append('ال')
        arabic_stopwords.append('اي')
        arabic_stopwords.append('ليه')
        arabic_stopwords.append('شركه')
        
        return arabic_stopwords

    def _Representation(self):

        representation_models = []
        if 'KeyBERT' in self.configs['representation_model_list']:
            key = KeyBERTInspired() 
            representation_models.append(key)
        
        if 'mmr' in self.configs['representation_model_list']:
            mmr = MaximalMarginalRelevance(diversity=self.configs['diversity'])
            representation_models.append(mmr)
        
        if 'zeroshot' in self.configs['representation_model_list']:
            zeroshot = ZeroShotClassification(self.configs['candidate_topics'], model=self.configs['zero_shot_model_name'])
            representation_models.append(zeroshot)

        return representation_models
    
    
    
