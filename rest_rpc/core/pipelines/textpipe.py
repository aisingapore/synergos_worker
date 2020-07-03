#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import math
import os
from pathlib import Path
from typing import Dict, List

# Libs
import contractions
import inflect
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import (
    TweetTokenizer, 
    regexp_tokenize, 
    sent_tokenize, 
    word_tokenize
)
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

# Custom

##################
# Configurations #
##################

SYMSPELL_WORD_PATH = "./inputs/nlp/frequency_dictionary_en_82_765.txt"
SYMSPELL_BIGRAM_PATH = "./inputs/nlp/frequency_bigramdictionary_en_243_342.txt"
MAX_EDIT_DISTANCE = 2

# Configure symspell for spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)
sym_spell.load_dictionary(SYMSPELL_WORD_PATH, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(SYMSPELL_BIGRAM_PATH, term_index=0, count_index=2)

#################### 
# Helper functions #
####################

def flatten(lst):
    """ Takes in a list of lists & combines them into 1 single list
    Args:
        lst: A list of lists
    Returns:
        Generator
    """
    for elem in lst:
        if type(elem) in (tuple, list):
            for i in flatten(elem):
                yield i
        else:
            yield elem


def truncate(number, digits) -> float:
    """ Truncates number to a specified number of digits
    Args:
        number (float): Number to truncate
        digits   (int): No. of digits to truncate to
    Returns:
        Truncated number (float)
    """
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

#######################################
# Data Preprocessing Class - TextPipe #
#######################################

class TextPipe:
    """
    The TextPipe class implement preprocessing tasks generalised for handling
    corpura. The general workflow is as follows:
    1) HTML tag removal
    2) Contraction expansion
    3) Number expansion
    4) Punctuation removal
    5) Spellchecking
    6) Lemmentization

    Prerequisite: Data MUST have its labels headered as 'target'

    Attributes:
        __seed (int): Seed to fix the random state of processes

        data   (): Loaded data to be processed
        output (pd.DataFrame): Processed data (with interpolations applied)
    """
    def __init__(self, data: List[str], seed=42):
        self.spellchecker = sym_spell
        self.data = data
        self.output = None

    ###########
    # Helpers #
    ###########

    def load_unified_corpus(self):
        """ Load all text datasets and combine them into a single corpus

        Returns:
            Unified corpus (pd.DataFrame)
        """
        all_loaded_corpus = [pd.read_csv(_path) for _path in self.data]
        
        self.output = pd.concat(
            all_loaded_corpus, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        self.output.columns = ['text', 'target']
        self.output['text'] = self.output['text'].str.split()

        return self.output


    def perform_spell_correction(self, word_set):
        """ Spell-checks a list of words & performs necessary corrections
        Args:
            word_set (list(str)): Words to be evaluated and corrected
        Returns:
            Corrected word set (list of strings)
        """
        # Combine all words in word set for bulk processing
        input_term = " ".join(word_set)
        # Retrieve bulk corrections
        suggestions = self.spellchecker.lookup_compound(
                        input_term, 
                        max_edit_distance=MAX_EDIT_DISTANCE,
                        transfer_casing=True)[0]._term
        # Split back into tokens
        corrected_word_set = suggestions.split(" ")
        return corrected_word_set


    def tokenize(self, data, convert_nums=False, spell_check=False):
        """ Convert an entire dataset into normalized tokens
            Normalisation process include:
                1) HTML tag removal
                2) Contraction expansion
                3) Number expansion
                4) Punctuation removal
                5) Spellchecking
                6) Lemmentization
        Args:
            data (list(str)): Dataset to be tokensized
        Returns:
            Words (list(str))
        """
        # Strip off all HTML tags
        no_html_tags = [BeautifulSoup(article, 'html.parser').get_text()
                        for article
                        in tqdm(data, desc=f"{'Removing HTML tags':<26}")]
        # Expand all contractions
        no_contractions = [contractions.fix(article)
                        for article
                        in tqdm(no_html_tags, 
                                desc=f"{'Expanding contractions':<26}")]
        # Fragment all articles into sentences
        sentences = [sent_tokenize(article)
                    for article
                    in tqdm(no_contractions, 
                            desc=f"{'Fragmenting into sentences':<26}")]
        # Fragment all sentences into lower-case words
        words = [[word.lower() 
                for word 
                in list(flatten(list(map(word_tokenize, s_set))))]
                for s_set 
                in tqdm(sentences, desc=f"{'Fragmenting into words':<26}")]
        # Remove all punctuations
        remove_punctuations = lambda word: re.sub(r'[^\w\s]', '', word)
        remove_blanks = lambda lst_of_words: [i for i in lst_of_words if i != ""]
        no_punctuations = [remove_blanks(list(map(remove_punctuations, w_set)))
                        for w_set 
                        in tqdm(words, desc=f"{'Removing punctuations':<26}")]
        # Replace all integer occurrences with textual representation
        no_numbers = no_punctuations
        if convert_nums:
            nums_to_char = lambda word: inflect.engine().number_to_words(word) \
                                        if word.isdigit() else word
            no_numbers = [list(map(nums_to_char, w_set)) for w_set in 
                        tqdm(no_punctuations, desc=f"{'Replacing numbers':<26}")]
        no_numbers = [[word for word in w_set if word is not re.search(r'\d+', word)]
                    for w_set 
                    in tqdm(no_punctuations, desc=f"{'Replacing numbers':<26}")]
        # Remove all stopwords 
        remove_stopwords = lambda w_set: [word for word in w_set \
                                        if word not in stopwords.words('english')] 
        no_stopwords = [remove_stopwords(w_set)
                        for w_set 
                        in tqdm(no_numbers, desc=f"{'Removing stopwords':<26}")]
        # Spellchecking
        no_wrong_words = no_stopwords
        if spell_check:
            no_wrong_words = [perform_spell_correction(w_set)
                            for w_set 
                            in tqdm(no_numbers, 
                                    desc=f"{'Spell-checking':<26}")]
        # Lemmatize all remaining words
        lemmatizer = WordNetLemmatizer()
        lemmatize_words = lambda w_set: [lemmatizer.lemmatize(w) for w in w_set]
        lemmatized = [lemmatize_words(w_set) 
                    for w_set
                    in tqdm(no_wrong_words, desc=f"{'Lemenatizing tokens':<26}")]
        return lemmatized


    def build_dictionary(self):
        pass

    ##################
    # Core functions #
    ##################

    def run(self) -> pd.DataFrame:
        """ Wrapper function that automates the text-specific preprocessing of
            the declared corpora

        Returns
            Output (pd.DataFrame) 
        """
        self.load_unified_corpus()

        return self.output 