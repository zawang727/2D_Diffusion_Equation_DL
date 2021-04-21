import importlib
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from keras.models import Sequential
from sklearn.model_selection import RepeatedKFold
from keras.models import model_from_json
from keras.layers import Dense
import tensorflow as tf
from keras import backend
import numpy as np
modulename='2d_diffusion_tool'

def get_model(Ninput,Noutput):
    model=Sequential()
    model.add(Dense(100,input_dim=Ninput,kernel_initializer='normal',activation='relu'))
    model.add(Dense(150,kernel_initializer='normal',activation='relu'))
    model.add(Dense(200,kernel_initializer='normal',activation='relu'))
    model.add(Dense(125,kernel_initializer='normal',activation='linear'))
    model.add(Dense(Noutput))
    model.compile(loss='mae',optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()
    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
    print(callbacks_list)
    return model    

def get_DL_data(train_input,train_output,test_input,test_output,file_path):
    iomodule=importlib.import_module(modulename)
    pfile=open(file_path,"r")
    oresult=iomodule.result_list()
    oresult=iomodule.result_reader(pfile,oresult)
    pfile.close()
    #  ML_model=LinearRegression()
    for i in range (0,int((len(oresult.data)-1)*0.8)):
        train_input.data.append(oresult.data[i].input.flatten())
        train_output.data.append(oresult.data[i].output.flatten())
    for i in range (int((len(oresult.data)-1)*0.8),len(oresult.data)-1):
        test_input.data.append(oresult.data[i].input.flatten())
        test_output.data.append(oresult.data[i].output.flatten())
    
def evaluate_model(X,y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
    cv = RepeatedKFold(n_splits=20, n_repeats=5, random_state=1);
    print(cv)
    # enumerate folds
    model = get_model(n_inputs, n_outputs)
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=200)
        # evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=0);
        # store result
        print(mae)
        results.append(mae)
    #print(results)
    return model

def save_model(model,model_name):
    model.save("model.h5_"+model_name)
    print("Saved model to disk")

def load_model(model_name):
    loaded_model=tf.keras.models.load_model('model.h5_'+model_name)
    print("Loaded model from disk")
    return loaded_model


print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
iomodule=importlib.import_module(modulename)
oresult_train_input=iomodule.result_list()
oresult_train_output=iomodule.result_list()
oresult_test_input=iomodule.result_list()
oresult_test_output=iomodule.result_list()
file_name="database.txt"
get_DL_data(oresult_train_input,oresult_train_output,oresult_test_input,oresult_test_output,file_name)
#NN_model=evaluate_model(np.asarray(oresult_train_input.data), np.asarray(oresult_train_output.data))
#save_model(NN_model,"30_30")
NN_model=load_model("30_30")
newX=np.asarray(oresult_test_input.data)
newY=NN_model.predict(newX)
iomodule.Two_D_Array_Contour_Show(iomodule.reconstruct_2D_array(newX[1]))
iomodule.Two_D_Array_Contour_Show(iomodule.reconstruct_2D_array(oresult_test_output.data[1]))
iomodule.Two_D_Array_Contour_Show(iomodule.reconstruct_2D_array(newY[1]))
#iomodule.Two_D_Array_Contour_Show(oresult_test_input.data[10])
#iomodule.Two_D_Array_Contour_Show(oresult_test_output.data[10])
#ML_model=get_model(len(oresult_train_input.data[0]),len(oresult_train_input.data))
#train data should be numpy not list
#ML_model.fit(np.asarray(oresult_train_input.data), np.asarray(oresult_train_output.data), verbose=0, epochs=100)
#mae = ML_model.evaluate(np.asarray(oresult_test_input.data), np.asarray(oresult_test_output.data), verbose=0)