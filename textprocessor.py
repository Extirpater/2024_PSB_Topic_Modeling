import pandas as pd
from unidecode import unidecode
from spellchecker import SpellChecker
import contractions
import re
import en_core_web_sm
import nltk
from nltk.corpus import stopwords
from ftfy import fix_encoding
import wordninja

class TextProcessor:
    def __init__(self):
        self.spell = SpellChecker()
        self.nlp = en_core_web_sm.load()
        additional_stop_words = {
            "conclusion", "method", "methodology",
            "results", "discussion", "figure", "figures", "tables", "appendix",
            "study", "research", "analysis", "paper", "findings", "data", "sample",
            "survey", "literature", "review", "case", "report", "experiment",   
            "theoretical", "practical", "implications", "model", "framework",
            "approach", "evidence", "hypothesis", "validation", "variables",
            "participants", "outcomes", "significant", "contribution", "limitations",
            "future", "work", "review", "experiment", "experimental", "design","pacific", "symposium", "biocomputing",
            "into", "three", "been", "study", "research", "supported", "where", "been", "may", "approach", 
        }

        self.stopwords = set(stopwords.words('english')).union(additional_stop_words)
          
    def remove_accent(self, text):
        return unidecode(text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def convert_lower(self, text):
        return text.lower()

    def special_figures(self, text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove symbols, digits, and special characters
        return text

    def remove_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def word_length(self, text):
        pattern = re.compile(r'(.)\1+')
        return pattern.sub(r'\1\1', text)
    
    def split_concatenated_words(self, text):
        words = wordninja.split(text)
        fixed_sentence = ' '.join(words)
        return fixed_sentence

    
    def remove_short_words(self, text):
        pattern = r'\b\w{1,2}\b'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def spell_correct(self, text):
        return ' '.join(self.spell.correction(word) for word in text.split() if self.spell.correction(word) is not None)

    def lemmatiz(self, txt):
        doc = self.nlp(txt)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text

    def remove_stopwords(self, text):
        return ' '.join(word for word in text.split() if word not in self.stopwords)
    
    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)
    
    def remove_cid(self, text):
        pattern = r'cid'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    
   
    def process_text(self, text, ra=True, ec=True, low=True, sf=True, rw=True, wl=True, lemma=True, sc=True, sw=True, digits=True, sws=True, cid=True):
        if ra: 
            text = self.remove_accent(text)
        if ec:
            text = self.expand_contractions(text)
        if low:
            text = self.convert_lower(text)
        if sf:
            text = self.special_figures(text)
        if digits:
            text = self.remove_numbers(text)
        if sw: 
            text = self.remove_stopwords(text)
        if rw:
            text = self.remove_whitespace(text)
        if wl: 
            text = self.word_length(text)
        if lemma: 
            text = self.lemmatiz(text)
        if sc: 
            text = self.spell_correct(text)
        if cid:
            text = self.remove_cid(text)
        text = fix_encoding(text)
        
        text = self.split_concatenated_words(text)
        if sws: 
            text = self.remove_short_words(text)
        return text

    def clean_text(self, text):
        text = re.sub(r"[^\w\s'-]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text_to_lines(self, text, words_per_line=10):
        words = text.split()
        lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
        return '\n'.join(lines)
