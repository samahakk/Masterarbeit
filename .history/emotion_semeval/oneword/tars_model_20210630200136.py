import flair 
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus

column_name_map = {0: "label", 1: "text"}

#1 get the corpus
corpus: Corpus = CSVClassificationCorpus(
                                   column_name_map=column_name_map,
                                   data_folder='../input/sentiment-amazon',
                                   test_file=None,
                                   skip_header=True,
                                   delimiter=','
) 

