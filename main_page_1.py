# Tensor version is 1.13.1
#d9103fae0133f3321d4a665ce76bd61cff95ba32

#kears 2.2.4
#NER+2 layers+crf+kfold+save model+use keras model via Keras_Model_CRF
# I split data in two part: train and test
#Then I k fold train data in two part: train and validation
#Hence, in this project, I I use train, validation for training and test for evalutiaon the model. So test set is constant and train and validation set is random
from sklearn.model_selection import KFold
from Create_LSTM_model_5 import CreatingModel
from Train_model_7 import Trainmodel
from Compile_model_6 import Compilemodel
from writing import Writelogs
from Loadrawdata_show_22_test import Loadrawdata_show_test
from Word_tags_to_idx_33_test import WordTagsToIdx_test
from Preprocessing_Sentence_to_idx_44_test import SentenceToIdx_test
from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
from Preprocessing_Sentence_to_idx_4 import SentenceToIdx_train
from Evaluate_model_8 import Evaluate_model
import numpy as np
from sklearn.model_selection import StratifiedKFold
filename_train='Raw_Data/Test_set.txt'
#filename_test='Raw_Data/Test_set.txt'
#Hyper paramet
n_fold=2
max_len_input=300
embedding_vector=50
epoch=50
bach_size=512
seed = 7
np.random.seed(seed)
unit1=embedding_vector
unit2=embedding_vector*2
#preparing TRAIN TEST
loadrawdata_train=Loadrawdata_show_train()
wrd_tag_vec_train=WordTagsToIdx_train()
wrd_tag_vec_train.word_tags_to_idx(loadrawdata_train.load(filename_train),max_len_input)
sent_idx_train=SentenceToIdx_train()
sent_idx_train.sentence_to_idx(wrd_tag_vec_train,max_len_input)

#preparing TEST SET
"""loadrawdata_test=Loadrawdata_show_test()
wrd_tag_vec_test=WordTagsToIdx_test()
wrd_tag_vec_test.word_tags_to_idx(loadrawdata_test.load(filename_test),max_len_input)
sent_idx_test=SentenceToIdx_test()
sent_idx_test.sentence_to_idx(wrd_tag_vec_test,max_len_input)"""
######################################
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
kfold.get_n_splits(sent_idx_train.X_train)
a=kfold
cvscores = []
namepic=1
#Neural Network
"""evaluation=Evaluate_model()
evaluation.evaluatemodel(sent_idx_test.X_test,sent_idx_test.Y_test, LSTM_implement.model_,wrd_tag_vec_train)"""
for train_index, validation_index in kfold.split(sent_idx_train.X_train):
    mywriting = Writelogs()
    print("------Start folder------------------")
    mywriting.writing(("---------------Start folder-----------------------") + '\n')
    print("TRAIN:", train_index, "TEST:", validation_index)
    print("----sent_idx_train.X_train[train_index]----:", sent_idx_train.X_train[train_index].shape, "-------y----:", sent_idx_train.Y_train[train_index].shape)
    # create model
    LSTM_implement = CreatingModel()
    LSTM_implement.creating_model(wrd_tag_vec_train, embedding_vector, unit1, unit2)
	# Compile model
    compile_model = Compilemodel()
    compile_model.compilemodel(LSTM_implement)
	# Fit the model
    train_model = Trainmodel()
    train_model.trainmodel(sent_idx_train.X_train[train_index], sent_idx_train.Y_train[train_index],sent_idx_train.X_train[validation_index], sent_idx_train.Y_train[validation_index], LSTM_implement.model_, epoch=epoch, batch_size=bach_size)
    # evaluate the model
    evaluation=Evaluate_model()
    #I use generated validation set in Preprcessing ot evaluate our model
    evaluation.evaluatemodel(sent_idx_train.X_test, sent_idx_train.Y_test, LSTM_implement.model_, wrd_tag_vec_train, cvscores)
    print("---------------End of  fold-----------------------")
    mywriting.writing(("---------------End of fold-----------------------") + '\n')
    namepic = 1+1





