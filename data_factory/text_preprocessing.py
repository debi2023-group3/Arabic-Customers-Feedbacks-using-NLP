# Importing Libraries

import re
import os
import string
import logging
import pandas as pd
from googletrans import Translator
 
SEED = 2023
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor(object):
    def __init__(self, df, args):
        """Initalization Function of TextPreprocessor class"""

        self.data = df
        self.args = args
        
    # ================================================================================= #
    def check_quality(self, verbose=False):
        # Display statistics before cleaning
        original_shape = self.data.shape
        if verbose:
            print(f"\t> Original Data Shape: {original_shape}")

        # Drop rows with missing values and remove duplicates
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        after_dup = self.data.shape
        if verbose:
            print(f"\t> Number of Duplicate Rows Removed: {original_shape[0] - after_dup[0]}")
        
        self.data = self.data.dropna().reset_index(drop=True)
        cleaned_shape = self.data.shape
        if verbose:
            print(f"\t> Number of NULLs Rows Removed: {after_dup[0] - cleaned_shape[0]}")
            print(f"\t> Cleaned Data Shape: {self.data.shape}")

        # Check data type of the target column and convert it to string if needed
        if self.data[self.args['text_col']].dtype != str :
            self.data.loc[:, self.args['text_col']] = self.data[self.args['text_col']].astype(str)

        # ================================================================================= #
    def detect_language_by_ascii(self, text):
        """ Detect language if it Arabic or English"""
        is_arabic = any(1536 <= ord(char) <= 1791 for char in text[:10])
        is_english = any(65 <= ord(char) <= 122 for char in text[:10])

        if is_arabic and not is_english:
            return "ar"
        elif is_english and not is_arabic:
            return "en"
        else:
            return "ar"
    # ================================================================================= #
    def translate_english(self):

        arabic_data = []
        tr = Translator()

        for idx, text in enumerate(self.data[self.args['text_col']]):
            lang = self.detect_language_by_ascii(text)
            if lang == "en":
                translated_text = tr.translate(text, src='en', dest='ar')
                arabic_data.append(translated_text.text)
            else:
                arabic_data.append(text)
        
        self.data[self.args['text_col']] = arabic_data
    # ================================================================================= #
    def save_file(self, path):
        os.makedirs(path, exist_ok=True)
        self.data.to_csv(f"{path}/processed_data.csv", index=False)
        logger.debug(f"Data saved in {path}/processed_data.csv")
    # ================================================================================= #
    def normalize_text(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
#             text = re.sub(" و ", " ", text)
#             text = re.sub("متاز","ممتاز", text)
#             text = re.sub("ممتاذ","ممتاز", text)
#             text = re.sub("خرة","فاشل", text)
#             text = re.sub("سئ","فاشل", text)
        text = re.sub(r"[ًًٌٍَُِّْ]", "", text) # Remove Tashkel   
        text = re.sub(r'(ـ+)', '', text) # Remove Tatwel  

        return text.strip()
    # ================================================================================= #
    def remove_stopwords(self, text):
        import arabicstopwords.arabicstopwords as stp

        arabic_stopwords = stp.stopwords_list()
        arabic_stopwords.remove('لا')
        arabic_stopwords.extend(['و','ليه','شركه','اي','تطبيقات','جد','تطبيق', 'برامج','ابلكيشن', 'برنامج', 'الي', 'ال'])
        tokens = text.split(" ")
        text = ' '.join([w for w in tokens if w not in arabic_stopwords])

        return text
    # ================================================================================= #
    def map_emojis(self, text):
        import json
        with open('./emojis_map.json', 'r') as f:
            emojis_map = json.load(f)
            logger.debug("Emojis mapper is loaded ...")

        for word in text:
            if word in emojis_map.keys():
                text = re.sub(word, emojis_map.get(word)+" ", text)
        
        self.remove_emojis(text)
        return text

    # ================================================================================= #   
    def remove_emojis(self, text):
        
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

        text = re.sub(emoji_pattern, '', text)
        
        return text
    # ================================================================================= #   
    def remove_punctuation(self, text):
        arabic_punctuations =  '''`÷×=؛<>_()*&^%][ـ،:".,'{}~¦+|!”…“–ـ/$£•●'''
        english_punctuations = string.punctuation
        numbers = "1234567890١٢٣٤٥٦٧٨٩٠"
        bad_characters = "�¿áóóó□"
        punctuations_list = arabic_punctuations + english_punctuations + numbers +  bad_characters

        replace_slash = str.maketrans('/', ' ', '')
        text = text.translate(replace_slash)
        remove_punc = str.maketrans('', '', punctuations_list)
        text = text.translate(remove_punc)

        return text
    # ================================================================================= #            
    def remove_latin(self, text):
        """ Removes any English word between the text """
        english_characters = re.compile(r'[a-zA-Z]')
        text = re.sub(english_characters, '', text)
        return text
    # ================================================================================= #            
    def remove_repeated_chars(self, text):
        """ Removes repeated characters """
        text = re.sub(r"(.)\1+", r"\1", text)
        return text
    # ================================================================================= #
    def remove_repeated_words(self, text):
        """ Removes repeated words """
        text = re.sub(r'\b(\w+)(?:\W+\1\b)+', r"\1", text)
        return text
    # ================================================================================= #
    def remove_long_words(self, text):
        """ Removes long words """
        tokens = [w for w in s.split(" ") if len(w) < 10]
        return " ".join(tokens)
    # ================================================================================= #
    def remove_url_email(self, text):
        """ Remove URLs """
        text = re.sub(r'http\S+|\S+@\S+', '', text)
        return text 
   # ================================================================================= #
    def stem_text(self, text):
        """ Stem each token in a text """
        from farasa.stemmer import FarasaStemmer
        stemmer = FarasaStemmer(interactive = True)
        text = stemmer.stem(text)
        return text  
    # ================================================================================= #
    def segment_text(self, text):
        """ Stem each token in a text """
        from farasa.stemmer import FarasaSegmenter
        stemmer = FarasaSegmenter(interactive = True)
        text = stemmer.segment(s)
        return text  
   
    # ================================================================================= #   
    def check_spelling(self, text):
        from spellchecker import SpellChecker
        checker = SpellChecker(language='ar')
        tokens = text.split(" ")
        
        correct_tokens = [checker.correction(token) for token in tokens]
        correct_tokens = [t if t is not None else tokens[i] for i, t in enumerate(correct_tokens)]
        
        if len(correct_tokens) > 1:
            return ' '.join(correct_tokens)
        else: 
            return text
        
    # ================================================================================= #
    def preprocess_text(self, text):

        if self.args['normalize']:
            text = self.normalize_text(text)
        if self.args['remove_url_email']:
            text = self.remove_url_email(text)
        if self.args['remove_punct']:
            text = self.remove_punctuation(text)
        if self.args['remove_repeated_chars']:
            text = self.remove_repeated_chars(text)
        if self.args['remove_repeated_words']:
            text = self.remove_repeated_words(text)
        if self.args['remove_long_words']:
            text = self.remove_long_words(text)
        if self.args['map_emojis']:
            text = self.map_emojis(text)
        if self.args['remove_emojis']:
            text = self.remove_emojis(text)
        if self.args['remove_english']:
            text = self.remove_latin(text)
        if self.args['remove_stopwords']:
            text = self.remove_stopwords(text)
        if self.args['check_spelling']:
            text = self.check_spelling(text)
        if self.args['stem']:
            text = self.stem_text(text)
        if self.args['segment']:
            text = self.segment_text(text)

        return text
    # ================================================================================= #
    def preprocess(self):
        """ map all the above functions to preprocess text """

        # 1. cleaning nulls and duplication
        self.check_quality(verbose=True)
        # 2. translate English samples
        print(">> Translating English Samples ...")
        if self.args['translate_english']:
            self.translate_english()
        # 3. Apply text preprocessing for NLP task
        print(">> Apply text preprocessing for NLP task ...")
        self.data.loc[self.args['text_col']] = self.data[self.args['text_col']].apply(self.preprocess_text)
        # 4. Check nulls and duplication after preprocessing 
        self.check_quality(verbose=True)
        # 5. save the cleaned and prepared file
        self.save_file(self.args['save_data_file'])
        
        return self.data
