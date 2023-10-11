import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

def evaluate_classifier(y_true, y_pred, classes): #, exp_name = 'classification'

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    # cr = classification_report(y_true, y_pred,labels=classes.values(),target_names=classes.keys(), zero_division=1)
    cm = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report':0.0,
        # 'roc_auc_micro': roc_auc_micro,
    }
    
    return metrics_dict



def calculate_coherence_score(topic_model, docs):
    # Preprocess documents
    cleaned_docs = topic_model._preprocess_text(docs)

    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    # depending on the version and if you get an error use commented out code below:
    # words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Create topic words
    topic_words = [[dictionary.token2id[w] for w in words if w in dictionary.token2id]
    for _ in range(topic_model.nr_topics)]

    # this creates a list of the token ids (in the format of integers) of the words in words that are also present in the 
    # dictionary created from the preprocessed text. The topic_words list contains list of token ids for each 
    # topic.

    coherence_model = CoherenceModel(topics=topic_words,
                                    texts=tokens,
                                    corpus=corpus,
                                    dictionary=dictionary,
                                    coherence='c_v')
    coherence = coherence_model.get_coherence()

    return coherence


def visualization(y_true, y_pred, probs, classes, save_dir = 'figures', task = 'Training'):

    os.makedirs(save_dir, exist_ok=True)

    # create and save visualization Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    cm_fig = plt.figure(figsize=(8, 6))
    cm_title = f"Confusion Matrix of {task}"
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes.values(),
                yticklabels=classes.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(cm_title)
    plt.savefig(os.path.join(save_dir,  cm_title + '.png'))

    # create and save visualization AUC
    y_true = label_binarize(y_true, classes=range(len(classes.keys()))) # np.eye(classes.keys())[y_true]
    probs = np.array(probs)
    
    # Micro-average AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    auc_fig = plt.figure(figsize=(8, 6))
    auc_title = f'Receiver Operating Characteristic (ROC) Curve of {task}'
    plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_micro:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(auc_title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir,  auc_title + '.png'))

    return cm_fig, auc_fig 


def save_results(metrics_dict, output_dir, task = 'Training'):
    
    if output_dir is None:
        raise ValueError("Please provide a `output_dir` for saving performance of the model.")
            
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "classifier_performance.txt")
    with open(file_path, 'a') as file:
        file.write(f"\n\n> Experiment Name: {task} \n")
        file.write(f"> Date: {datetime.now()}\n")
        file.write(f"> Performance: \n")
        for metric, value in metrics_dict.items():
            file.write(f"- {metric}: \n{value}\n")
        file.write('-'*50)

    return file_path