from flair.data import Corpus
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus

column_name_map = {0: "label", 1: "text"}

neword'
#1 get the corpus
corpus: Corpus = CSVClassificationCorpus(
                                   column_name_map=column_name_map,
                                   data_folder=data_folder,
                                   test_file=None,
                                   skip_header=True,
                                   delimiter=','
) 

tars = TARSClassifier(task_name='emotion_semeval_oneword', label_dictionary=corpus.make_label_dictionary())

trainer = ModelTrainer(tars, corpus)

# 5. start the training
# 5. start the training
trainer.train(base_path='resources/taggers/emotion_semeval_oneword', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )