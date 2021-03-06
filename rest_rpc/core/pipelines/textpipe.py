#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import concurrent.futures
import math
import os
import re
from collections import Counter
from logging import NOTSET
from pathlib import Path
from typing import Dict, List

# Libs
import contractions
import inflect
import nltk
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

# Custom
from rest_rpc import app
from rest_rpc.core.pipelines.base import BasePipe
from rest_rpc.core.pipelines.dataset import PipeData

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

# Configure symspell for spelling correction
symspell_dictionaries = app.config['SYMSPELL_DICTIONARIES']
symspell_bigrams = app.config['SYMSPELL_BIGRAMS']
MAX_EDIT_DISTANCE = 2

sym_spell = SymSpell(
    max_dictionary_edit_distance=MAX_EDIT_DISTANCE, 
    prefix_length=7
)
for ssp_dict_path in symspell_dictionaries:
    sym_spell.load_dictionary(ssp_dict_path, term_index=0, count_index=1)

for ssp_bigram_path in symspell_bigrams:
    sym_spell.load_dictionary(ssp_bigram_path, term_index=0, count_index=2)

# Configure Spacy for nlp operations
# Note: For the current set of supported NLP operations, Spacy implementations 
# have been tested to be computationally worse off than NLTK implementations. 
# Hence, temporarily disable Spacy loadings to exclude Spacy-related runtime 
# errors.
# spacy_nlp = spacy.load('en_core_web_sm')

cores_used = app.config['CORES_USED']

SEED = 42

logging = app.config['NODE_LOGGER'].synlog
logging.debug("custom.py logged", Description="No Changes")

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

class TextPipe(BasePipe):
    """
    The TextPipe class implement preprocessing tasks generalised for handling
    corpura. The general workflow is as follows:

    1) HTML tag removal
    2) Contraction expansion
    3) Stopword removal
    4) Number expansion
    5) Punctuation removal
    6) Spellchecking
    7) Lemmentization

    Prerequisite: Data MUST have its labels headered as 'target'

    Attributes:
        data   (): Loaded data to be processed
        output (pd.DataFrame): Processed data (with interpolations applied)
    """

    def __init__(
        self, 
        data: List[str],
        des_dir: str,
        max_df: int = 30000,
        max_features: int = 1000,
        strip_accents: str = 'unicode',
        keep_html: bool = False,
        keep_contractions: bool = False,
        keep_punctuations: bool = False,
        keep_numbers: bool = False,
        keep_stopwords: bool = False,
        spellcheck: bool = True,
        lemmatize: bool = True,
    ):
        super().__init__(datatype="text", data=data, des_dir=des_dir)

        self.__spellchecker = sym_spell

        self.max_df = max_df
        self.max_features = max_features
        self.strip_accents = strip_accents
        self.keep_html = keep_html
        self.keep_contractions = keep_contractions
        self.keep_punctuations = keep_punctuations
        self.keep_numbers = keep_numbers
        self.keep_stopwords = keep_stopwords
        self.spellcheck = spellcheck
        self.lemmatize = lemmatize

    ###########
    # Helpers #
    ###########

    def load_unified_corpus(self):
        """ Load all text datasets and combine them into a single corpus

        Returns:
            Unified corpus (pd.DataFrame)
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_loaded_corpus = list(executor.map(pd.read_csv, self.data))
        
        unified_corpus = pd.concat(
            all_loaded_corpus, 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        unified_corpus.columns = ['text', 'target']

        return unified_corpus


    def create_docterm_matrix(self, df):
        """ Converts a dataframe of articles into a count matrix in preparation
            for training or inference.

            IMPORTANT:
            Specified dataframe has to have a `target` column with the label
            classifications of the dataset at hand

        Args:
            df (pd.DataFrame): Dataframe of articles
        Returns:
            Word Vector Dataframe (pd.DataFrame)
        """
        # Convert dataset into bag of words
        vectorizer = CountVectorizer(
            strip_accents=self.strip_accents,
            max_df=self.max_df,
            max_features=self.max_features
        )
        doc_term_matrix = vectorizer.fit_transform(df['text'].values).toarray()
        vocabulary = vectorizer.get_feature_names()

        # Reformat outputs to dataframe
        word_vector_df = pd.DataFrame(data=doc_term_matrix, columns=vocabulary)
        word_vector_df['target'] = df['target'].astype('category')

        return word_vector_df


    @staticmethod
    def strip_html_tags(articles: List[str]) -> List[str]:
        """ Remove all html tags from each article declared
        
        Args:
            articles (list(str)): Articles to be evaluated and corrected
        Returns:
            Corrected list of articles
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            no_html_tags = list(tqdm(
                executor.map(
                    lambda x: BeautifulSoup(x, 'html.parser').get_text(),
                    articles
                ),
                total=len(articles),
                desc=f"{'Removing HTML tags':<26}"
            ))

        return no_html_tags


    @staticmethod
    def expand_contractions(articles: List[str]) -> List[str]:
        """ Expand all contractions in all declared articles
        
        Args:
            articles (list(str)): Articles to be evaluated and corrected
        Returns:
            Corrected list of articles (list(str))
        """       
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            no_contractions = list(tqdm(
                executor.map(contractions.fix, articles),
                total=len(articles),
                desc=f"{'Expanding contractions':<26}"
            ))

        return no_contractions


    @staticmethod
    def fragment_into_sentences(articles: List[str]) -> List[List[str]]:
        """ Fragment all declared articles into their respective sentences

        Args:
            articles (list(str)): Articles to be evaluated and corrected
        Returns:
            Corrected list of sentence lists
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:

            sentences = list(tqdm(
                executor.map(sent_tokenize, articles), # list of sentences
                total=len(articles),
                desc=f"{'Fragmenting into sentences':<26}"
            ))
            
        return sentences


    @staticmethod
    def fragment_into_words(sentence_lists: List[List[str]]) -> List[List[str]]:
        """ Fragment all declared sentence lists into lower-case words sets

        Args:
            sentence_lists (list(list(str))): list of sentences to be evaluated
        Returns:
            Word sets (list(list(str)))
        """ 
        extract_words = lambda list_of_sentences: [
            word.lower() 
            for word 
            in list(flatten(list(map(word_tokenize, list_of_sentences))))
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:

            word_sets = list(tqdm(
                executor.map(extract_words, sentence_lists), # list of sentences
                total=len(sentence_lists),
                desc=f"{'Fragmenting into words':<26}"
            ))
            
        return word_sets


    # @staticmethod
    # def fragment_into_words(articles: List[str]) -> List[List[str]]:
    #     """ Fragment all declared sentence lists into lower-case words sets

    #     Args:
    #         articles (list(str)): list of sentences to be evaluated
    #     Returns:
    #         Word sets (list(list(str)))
    #     """ 
    #     tokenize_article = lambda x: [token for token in spacy_nlp(x)]
    #     with concurrent.futures.ThreadPoolExecutor() as executor:

    #         word_sets = list(tqdm(
    #             executor.map(tokenize_article, articles),
    #             total=len(articles),
    #             desc=f"{'Fragmenting into words':<26}"
    #         ))

    #     logging.debug(f"word sets: {word_sets}")
    #     return word_sets


    @staticmethod
    def remove_punctuations(word_sets: List[List[str]]) -> List[List[str]]:
        """ Remove all punctuations from all declared word sets

        Args:
            word_sets (list(list(str))): Words to be evaluated and corrected
        Returns:
            Corrected word sets (list(list(str)))
        """
        remove_punctuations = lambda word: re.sub(r'[^\w\s]', '', word)
        remove_blanks = lambda word_set: [i for i in word_set if i != ""]
        with concurrent.futures.ThreadPoolExecutor() as executor:

            no_punctuations = list(tqdm(
                executor.map(
                    lambda x: remove_blanks(list(map(remove_punctuations, x))), 
                    word_sets
                ),
                total=len(word_sets),
                desc=f"{'Removing punctuations':<26}"
            ))
            
        return no_punctuations


    @staticmethod
    def convert_numbers(word_sets: List[List[str]]) -> List[List[str]]:
        """ Replace all integer occurrences with textual representation for all
            declared word sets

        Args:
            word_sets (list(list(str))): Words to be evaluated and corrected
        Returns:
            Corrected word sets (list(list(str)))
        """
        nums_to_char = lambda word: inflect.engine().number_to_words(word) \
                                    if word.isdigit() else word
        with concurrent.futures.ThreadPoolExecutor() as executor:

            converted_numbers = list(tqdm(
                executor.map(
                    lambda x: list(map(nums_to_char, x)), 
                    word_sets
                ),
                total=len(word_sets),
                desc=f"{'Replacing numbers':<26}"
            ))
            
        return converted_numbers
        
        
    @staticmethod
    def remove_numbers(word_sets: List[List[str]]) -> List[List[str]]:
        """ Remove numbers from all declared word sets

        Args:
            word_sets (list(list(str))): Words to be evaluated and corrected
        Returns:
            Corrected word sets (list(list(str)))
        """
        prune_remaining_numbers = lambda w_set: [
            word 
            for word in w_set 
            if word is not re.search(r'\d+', word)
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:

            no_numbers = list(tqdm(
                executor.map(
                    prune_remaining_numbers, 
                    word_sets
                ),
                total=len(word_sets),
                desc=f"{'Removing residual numbers':<26}"
            ))
                
        return no_numbers


    @staticmethod
    def remove_stopwords(word_sets: List[List[str]]) -> List[List[str]]:
        """ Remove all stopwords from all declared word sets

            IMPORTANT:
            This function is unable to parallelize properly, because the wordnet
            is still a proxy object when the second thread first accesses it 
            despite having had its class and dict changed. Hence, when the 
            machine is loading the corpus slowly, it stalls all other proxy 
            objects while they too try to load the corpus
            
        Args:
            word_sets (list(list(str))): Words to be evaluated and corrected
        Returns:
            Corrected word sets (list(list(str)))
        """ 
        no_stopwords = [
            [
                word 
                for word in w_set
                if word not in stopwords.words('english')
            ]
            for w_set
            in tqdm(word_sets, desc=f"{'Removing stopwords':<26}")
        ]
            
        return no_stopwords


    def correct_spelling(self, word_sets: List[List[str]]) -> List[List[str]]:
        """ Spellchecks and corrects all words found within declared word sets
    
        Args:
            word_sets (list(list(str))): Words to be evaluated and corrected
        Returns:
            Corrected word sets (list(list(str)))
        """
        def perform_spell_correction(word_set):
            """ Spell-checks a list of words & performs necessary corrections

            Args:
                word_set (list(str)): Words to be evaluated and corrected
            Returns:
                Corrected word set (list of strings)
            """
            # Combine all words in word set for bulk processing
            input_term = " ".join(word_set)

            # Retrieve bulk corrections
            suggestions = self.__spellchecker.lookup_compound(
                input_term, 
                max_edit_distance=MAX_EDIT_DISTANCE,
                transfer_casing=True
            )[0]._term

            # Split back into tokens
            corrected_word_set = suggestions.split(" ")
            return corrected_word_set

        # Attempts to parallelise this operation have failed to improve
        # pre-processing speed. Hence, using vanilla `for` loops since it has
        # even better performance as compared to `ThreadPoolExecutor()` or 
        # 'ProcessPoolExecutor()`!

        corrected_words = [ 
            perform_spell_correction(w_set)
            for w_set
            in tqdm(word_sets, desc=f"{'Spell-checking':<26}")
        ]
        return corrected_words


    @staticmethod
    def lemmatize_words(word_sets: List[List[str]]) -> List[List[str]]:
        """ Lemmatize words for all words found within declared word sets

            IMPORTANT:
            This function is unable to parallelize properly, because the wordnet
            is still a proxy object when the second thread first accesses it 
            despite having had its class and dict changed. Hence, when the 
            machine is loading the corpus slowly, it stalls all other proxy 
            objects while they too try to load the corpus

        Args:
            word_set (list(str)): Words to be evaluated and corrected
        Returns:
            Corrected word set (list of strings)
        """
        lemmatizer = WordNetLemmatizer()
        lemmatize_words = lambda w_set: [lemmatizer.lemmatize(w) for w in w_set]

        lemmatized_wordsets = [ 
            lemmatize_words(w_set)
            for w_set
            in tqdm(word_sets, desc=f"{'Lemenatizing tokens':<26}")
        
        ]

        return lemmatized_wordsets


    @staticmethod
    def retrieve_doc_representation(doc_idx, dtm, vocab):
        """ Retrieves document frequency mapping of specified document
        Args:
            doc_idx         (int): Index of document
            dtm (np.ndarray(int)): Document-term matrix of corpus
            vocab (np.array(str)): Vocabulary set of corpus
        Returns:
            Document frequency mapping (dict(str, int))
        """
        documents = pd.DataFrame(data=dtm, columns=vocab)
        doc = documents.loc[doc_idx, (documents.loc[doc_idx, :] > 0)]
        words = doc.index
        return Counter(dict(zip(words, doc)))


    @staticmethod
    def retrieve_topics(w_matrix, vocabulary, n_top_words):
        """ Find topics using LDA given a pre-computed word vector matrix 
        Args:
            w_matrix   (np.ndarray): Word vector matrix of the current corpus
            vocabulary (np.ndarray): Possible words in corpus
            n_top_words       (int): No. of words to represent topics
        Returns:
            All topics found (dict(int, str))
        """
        lda = LatentDirichletAllocation(
            n_components=10, 
            learning_method='batch',
            learning_decay=0.7, 
            learning_offset=10.0, 
            max_iter=10, 
            evaluate_every=-1, 
            perp_tol=0.1, 
            mean_change_tol=0.001, 
            max_doc_update_iter=100, 
            n_jobs=-1,
            random_state=SEED
        )
        lda.fit(w_matrix)
        topics = {}
        for t_idx, topic in enumerate(lda.components_):
            reverse_sort_top_n = topic.argsort()[::-1][:n_top_words]
            topic_words = [vocabulary[i] for i in reverse_sort_top_n]
            topics[t_idx+1] = topic_words
        return topics

    ##################
    # Core functions #
    ##################

    def run(self) -> pd.DataFrame:
        """ Wrapper function that automates the text-specific preprocessing of
            the declared corpora. Optional normalisation processes include:

            1) HTML tag removal
            2) Contraction expansion
            3) Stopword removal
            4) Number expansion
            5) Punctuation removal
            6) Spellchecking
            7) Lemmentization

        Returns
            Output (pd.DataFrame) 
        """
        if not self.is_processed():
            unified_corpus = self.load_unified_corpus()

            articles = unified_corpus['text'].tolist()

            ######################################
            # Stage 1: Document-level operations #
            ######################################
            if not self.keep_html:
                articles = self.strip_html_tags(articles)

            if not self.keep_contractions:
                articles = self.expand_contractions(articles)

            ##################################
            # Stage 2: Word-level operations #
            ##################################
            sentence_lists = self.fragment_into_sentences(articles)
            word_sets = self.fragment_into_words(sentence_lists)

            # word_sets = self.fragment_into_words(articles)

            if not self.keep_punctuations:
                word_sets = self.remove_punctuations(word_sets)

            if not self.keep_numbers:
                word_sets = self.convert_numbers(word_sets)
                word_sets = self.remove_numbers(word_sets)

            if not self.keep_stopwords:
                word_sets = self.remove_stopwords(word_sets)

            if self.spellcheck:
                word_sets = self.correct_spelling(word_sets)

            if self.lemmatize:
                word_sets = self.lemmatize_words(word_sets)

            # Re-combine remaining words into a token article
            tokenised_articles = [" ".join(w_set) for w_set in word_sets]
            unified_corpus['text'] = tokenised_articles

            self.output = self.create_docterm_matrix(unified_corpus)

        logging.log(
            level=NOTSET,
            event="Doc-term matrix tracked.",
            docterm_matrix=self.output, 
            ID_path=SOURCE_FILE,
            ID_class=TextPipe.__name__, 
            ID_function=TextPipe.run.__name__
        )

        return self.output