import pandas as pd
import numpy as np
from tsfresh import select_features

import random

from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings("ignore")

random.seed(123)

#to change with the actual class -- gender / age
mapper = {'p02-old': 1, 'p03-old': 1, 'p04-old': 0, 'p05-old': 0, 'p06-old': 1, 'p08-old': 1,
'p09-old': 1, 'p10-old': 0, 'p11-old': 0, 'p12-old': 0, 'p13-old': 0, 'p14-old': 0,
'p15-old': 0, 'p01-young': 0, 'p02-young': 0, 'p03-young': 1, 'p04-young': 1, 'p05-young': 1,
'p07-young': 1, 'p08-young': 1, 'p09-young': 0,'p10-young': 0, 'p11-young': 1, 'p12-young': 0,
'p13-young': 1, 'p14-young': 1, 'p15-young': 1, 'p17-young': 0, 'p19-old': 1, 'p20-young': 0,
'p21-young': 0, 'p22-young': 1, 'p23-young': 0, 'p24-young': 1, 'p25-young': 0}

#from sets import Set
from random import sample

def get_statistica_set(s):
    diz = {}

    for i in s:
        if  not i in diz:
            diz[i] = 1   #creare la chiave i con valore 1           
        else:
            diz[i] +=1   #increemntare valore chiave
            
    print("distruzione:", diz)
   

def random_partition(n,k,j, lista) :
    #print(lista)
    ls1=[]  #ne prende n elementi
    ls2=[]   #ne prende k elementi
    ls3=[]  # ne prende j elementi
    
    ls1=sample(list(lista),n)
    #print(ls1)
    
    non_presi=set(lista)-set(ls1)  #gli elementi ancora liberi non presi 
    ls2=sample(non_presi,k)
    #
    ls3=set(lista)-set(ls1)-set(ls2)
    return ls1,ls2,ls3

def buid_datafram_from_user(ls):
    out_Dataframe = pd.DataFrame()   #un datafram vuoto
    #prova=["dennymiche","7efq4p950y62qep4257bpy7ks"]

    for u in ls:                        #concatenate tutte le sub datafram che corrispondono agli user
        u_frame= df_start[df_start['id_owner'] ==u ]
        frames = [out_Dataframe, u_frame]
        out_Dataframe =pd.concat(frames)
    return out_Dataframe


def get_model(algorithm):   #return un modello con vari parametri

    param_grid_dt ={
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 3, 5]
    }

    param_grid_rf ={
        'n_estimators': [25, 50, 100],
        'max_depth': [ 3, 5, 7],
        'min_samples_leaf': [1, 3, 5]
    }

    param_grid_lr = {
        'C' : [0.1,1,10],
        'max_iter' : [10000]
    }

    param_grid_la = {
        'alpha': [0.01, 0.1, 1., 10],
        'fit_intercept': [False, True],
        'max_iter' : [10000]
    }
    if algorithm == "RF":
        clf = RandomForestClassifier(random_state=123)
        param_grid = param_grid_rf
    if algorithm == "LR":
        clf = LogisticRegression(random_state=123)
        param_grid = param_grid_lr
    if algorithm == "DT":
        clf = DecisionTreeClassifier(random_state=123)
        param_grid = param_grid_dt
    if algorithm == 'RI':
        clf = RidgeClassifier(random_state = 123)
        param_grid = param_grid_la

    return clf, param_grid


def genera_set_train_val_test(colonna_target):

    
    #divisione in X e Y : feature e target 
    
    index_target=df_start.columns.get_loc(colonna_target)  #usare questo per trovare index :)
                                                        #voglio predire extroversion colonna 81; 60=occupation; 61 sport ! 
                                                        
    #train 
    train_data = train_dataframe.values
    #print(train_data)
    # m = number of input samples
    m_train = train_data.shape[0]
    print("Amount of data ready for train:",m_train)



    Y_train = train_data[:m_train,index_target]   #voglio predire extroversion colonna 81; 60=occupation; 61 sport ! 
    print("target",Y_train)



    X_train = train_data[:m_train,8:53]       #prende da 0-m come numero di riga; e dalla 8 colonna in poi come COLONNA
    feature_names = train_dataframe.columns[8:53]


    print("feature",feature_names)


    #######################################################################
    #val 
    val_data = val_dataframe.values
    #print(train_data)
    # m = number of input samples
    m_val = val_data.shape[0]
    print("Amount of data ready for val:",m_val)
    
    Y_val = val_data[:m_val,index_target]   #se voglio predire economics che è in riga 62
    #print("target age",Y)
    
    
    X_val = val_data[:m_val,8:53]       #prende da 0-m come numero di riga; e dalla 4 colonna in poi come COLONNA
    feature_names =val_dataframe.columns[8:53]
    #print("feature",feature_names)
    
    ########################################################################
    #test
    #val 
    test_data = test_dataframe.values
    #print(train_data)
    # m = number of input samples
    m_test = test_data.shape[0]
    print("Amount of data ready for test:",m_test)
    
    Y_test = test_data[:m_test,index_target]   #voglio predire age che è in riga 56; economcis 62
    #print("target age",Y)
    
    
    X_test = test_data[:m_test,8:53]       #prende da 0-m come numero di riga; e dalla 4 colonna in poi come COLONNA
    feature_names = test_dataframe.columns[8:53]
    #print("feature",feature_names)
    
    #mettere as int
    Y_train=Y_train.astype('int')
    Y_test=Y_test.astype('int')
    Y_val=Y_val.astype('int')
    
    return X_train,Y_train,X_val,Y_val,X_test, Y_test

def genera_set_train_val_test_Con_Correlated_Feature(colonna_target,ls): #viene passato il target e le best feature coorelate 
    #divisione in X e Y : feature e target 
    
    
    #IMPORTANTE : faccio un sub DATAFRAMe con target INTESSATO e in prima colonna e le best SET FEATURE IN SEGUITO
   
    ls.insert(0, colonna_target)
    best_train_feature_Datafram_with_target=train_dataframe[ls]
    
    
    #["n_follower_playlist","numero_brani","anno_min","anno_max","average_anno_publicazione","std_anno_pubblicazione","n_brani_solo_perform","n_brani_COLLAB_perform","un artista compare multi time:","max_time","simpson_index","ratio_M_artist","ratio_f_artist","N artisti non popular(fans <1000)","avg_popularity_songs","std_popularity_songs","Is playlist updated","perc_pop","perc_hiphop","perc_rap","perc_rock","perc_edm","perc_latin","perc_indie","perc_classic","perc_kpop","perc_metal","perc_country","danceability","std_danceability","energy","energy_std","loudness","loudness_std","speechiness","speechiness_std","acoutsticness","acoutsticness_std","instrumentalness","instrumentalness_std","liveness","liveness_std","valence","valence_std","tempo","tempo_std"]
    
    train_data = best_train_feature_Datafram_with_target.values
    #print(train_data)
    # m = number of input samples
    m_train = train_data.shape[0]   #len
    print("Amount of data ready for train:",m_train)
    
    Y_train = train_data[:m_train,0]   #voglio predire age che è in poszione 0 da come ho costruito io
    print("target",Y_train)
    
    
    
    X_train = train_data[:m_train,1:]       #prende da 0-m come numero di riga; e dalla 1 colonna in poi come COLONNA
    #print(type(X_train))
    feature_names = best_train_feature_Datafram_with_target.columns[1:]
    
    
    print("feature",feature_names)
    
    #######################################################################
    #IMPORTANTE : faccio un sub DATAFRAMe con target age in prima colonna e le best SET FEATURE IN SEGUITO
    best_val_feature_Datafram_with_target= val_dataframe[ls]
    
    #["n_follower_playlist","numero_brani","anno_min","anno_max","average_anno_publicazione","std_anno_pubblicazione","n_brani_solo_perform","n_brani_COLLAB_perform","un artista compare multi time:","max_time","simpson_index","ratio_M_artist","ratio_f_artist","N artisti non popular(fans <1000)","avg_popularity_songs","std_popularity_songs","Is playlist updated","perc_pop","perc_hiphop","perc_rap","perc_rock","perc_edm","perc_latin","perc_indie","perc_classic","perc_kpop","perc_metal","perc_country","danceability","std_danceability","energy","energy_std","loudness","loudness_std","speechiness","speechiness_std","acoutsticness","acoutsticness_std","instrumentalness","instrumentalness_std","liveness","liveness_std","valence","valence_std","tempo","tempo_std"]
    
    val_data = best_val_feature_Datafram_with_target.values
    #print(val_data)
    # m = number of input samples
    m_val = val_data.shape[0]   #len
    print("Amount of data ready for val:",m_val)
    
    Y_val = val_data[:m_val,0]   #voglio predire age che è in poszione 0 da come ho costruito io
    #print("target ",Y_val)
    
    
    
    X_val = val_data[:m_val,1:]       #prende da 0-m come numero di riga; e dalla 1 colonna in poi come COLONNA
    #print(type(X_val))
    feature_names = best_val_feature_Datafram_with_target.columns[1:]
    
    
    #print("feature",feature_names)
    #######################################################################
    
    #IMPORTANTE : faccio un sub DATAFRAMe con target age in prima colonna e le best SET FEATURE IN SEGUITO
    best_test_feature_Datafram_with_target= test_dataframe[ls]
    
    #["n_follower_playlist","numero_brani","anno_min","anno_max","average_anno_publicazione","std_anno_pubblicazione","n_brani_solo_perform","n_brani_COLLAB_perform","un artista compare multi time:","max_time","simpson_index","ratio_M_artist","ratio_f_artist","N artisti non popular(fans <1000)","avg_popularity_songs","std_popularity_songs","Is playlist updated","perc_pop","perc_hiphop","perc_rap","perc_rock","perc_edm","perc_latin","perc_indie","perc_classic","perc_kpop","perc_metal","perc_country","danceability","std_danceability","energy","energy_std","loudness","loudness_std","speechiness","speechiness_std","acoutsticness","acoutsticness_std","instrumentalness","instrumentalness_std","liveness","liveness_std","valence","valence_std","tempo","tempo_std"]
    
    test_data = best_test_feature_Datafram_with_target.values
    #print(test_data)
    # m = number of input samples
    m_test = test_data.shape[0]   #len
    print("Amount of data ready for test:",m_test)
    
    Y_test = test_data[:m_test,0]  #voglio predire age che è in poszione 0 da come ho costruito io
    #print("target ",Y_test)
    
    
    
    X_test = test_data[:m_test,1:]       #prende da 0-m come numero di riga; e dalla 1 colonna in poi come COLONNA
    #print(type(X_test))
    feature_names = best_test_feature_Datafram_with_target.columns[1:]
    
    
    #print("feature",feature_names)
    
    return X_train,Y_train,X_val,Y_val,X_test, Y_test
    
############################################################################################

if __name__ == '__main__':
    #read the dataset with the extracted features
    df_start = pd.read_csv("output_concatenate.csv", index_col = "Unnamed: 0")
    #print(df_start["id_owner"])

    
    #df_start["user"] = [x.split("|")[1] for x in df_start.index]
    res_df = pd.DataFrame(None, columns = ["#iter", "model", "task", "sensor", "f1"])
    
    ls_user=df_start["id_owner"].unique()
    print("In totale ci sono user distinti",len(ls_user))
    
    n_user_tot=len(ls_user)
    n_user_train = int(7./10.*n_user_tot)
    n_user_val=int(1./10.*n_user_tot)
    n_user_test=n_user_tot-n_user_train-n_user_val
    print("Number User for training and deciding parameters:",n_user_train)
    print("Number User for validation (choosing among different models):",n_user_val)
    print("Number User for test:",n_user_test)
    
    #list users : prende 70% user per train 10% user per validation e 20% per test
    user_train , user_val , user_test = random_partition(n_user_train,n_user_val,n_user_test,ls_user)  # do in input ls user totale e quanti ne deve prednere 

    #costruisco i datafram di train ,validation e test con le loro plyalist! 
    train_dataframe = buid_datafram_from_user(user_train) # costruisco un frame con solo user decisi prima ! 
    val_dataframe= buid_datafram_from_user(user_val)
    test_dataframe=buid_datafram_from_user(user_test)



    print("\nNumber train_Set playlist: "+str(len(train_dataframe))+" and belong to "+str(len(train_dataframe["id_owner"].unique()))+" users")
    print("Number val_Set playlist: "+str(len(val_dataframe))+" and belong to "+str(len(val_dataframe["id_owner"].unique()))+" users")
    print("Number test_Set playlist: "+str(len(test_dataframe))+" and belong to "+str(len(test_dataframe["id_owner"].unique()))+" users")
    
    #TOGLIERE I NAN 1!!!!!!!!!
    print("\nTolgo i Nan\n")
    train_dataframe= train_dataframe.dropna() 
    test_dataframe= test_dataframe.dropna() 
    val_dataframe= val_dataframe.dropna() 
    
    #print(train_dataframe.isnull().values.any())
    #print(val_dataframe.isnull().values.any())
    #print(test_dataframe.isnull().values.any())
    
    
    #Costruisco i Xtrain,Xval,Xtest , Ytrain,Yval, Ytest
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = genera_set_train_val_test("How do you consider your Economic Status? ")
    #avere qualche statistica
    get_statistica_set(Y_train)
    # Data pre-processing
    from sklearn import preprocessing
    print("\nscaler dataset...")
    scaler = preprocessing.StandardScaler().fit(X_train)
    Xtrain_scaled = scaler.transform(X_train)
    Xval_scaled = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)
    print("done scaled")
        
    
    
    print("\nbuild ora uno split per grid-search")
    #build the split for grid-search
    
    #print(len(Xtrain_scaled))
    #print("____________________")
    #print(len(Xval_scaled))
    
    X_train_val_scaled=np.concatenate((Xtrain_scaled,Xval_scaled), axis=0)
    Y_train_val=np.concatenate((Y_train,Y_val), axis=0)
    #print("ecco concatenato")
    #print(len(X_train_val_scaled))
    #print(len(Y_train_val))
    print("done")
    
    #============ Dummy ============

    algorithm = "Dummy"
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(Xtrain_scaled,Y_train)
    y_dummy = dummy_clf.predict(X_test)
    score = f1_score(Y_test, y_dummy, average="macro")
    print("\nDummy model f1 score",score)
    
    #============ Logistic Regression =============
    algorithm = "LR"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Logistica Regression",score)
    
    #============ Decision Tree =============
    algorithm = "DT"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Decision Tree",score)
    
    #============ Ridged classificer =============
    algorithm = "RI"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Ridged classificer",score)
    
    
    
    
    #con smote 
    
    print("\nCON SMOTE ")
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, Y_train)
    #print(X_resampled)
    #print(y_resampled)
    get_statistica_set(y_resampled)
    
    print("\nscaler smote dataset ...")
    scaler = preprocessing.StandardScaler().fit(X_resampled)
    Xtrain_scaled = scaler.transform(X_resampled)
    Xval_scaled = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)
    print("done scaled smote")
    print("\nbuild ora uno split per grid-search")
    
    
    X_train_val_scaled=np.concatenate((Xtrain_scaled,Xval_scaled), axis=0)
    Y_train_val=np.concatenate((y_resampled,Y_val), axis=0)
    #print("ecco concatenato")
    #print(len(X_train_val_scaled))
    #print(len(Y_train_val))
    print("done")
    
    #============ Dummy ============

    algorithm = "Dummy"
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(Xtrain_scaled,y_resampled)
    y_dummy = dummy_clf.predict(X_test)
    score = f1_score(Y_test, y_dummy, average="macro")
    print("\nDummy model f1 score",score)
    
    #============ Logistic Regression =============
    algorithm = "LR"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Logistica Regression",score)
    
    #============ Decision Tree =============
    algorithm = "DT"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Decision Tree",score)
    
    #============ Ridged classificer =============
    algorithm = "RI"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Ridged classificer",score)
    
    
    
    
    
    print("=====================================ORA CON LE BEST FEATURE=====================================")
    best_feature=["std_anno_pubblicazione","n_brani_solo_perform","perc_metal","energy","acoutsticness"]
    X_train, Y_train, X_val, Y_val, X_test, Y_test = genera_set_train_val_test_Con_Correlated_Feature("How do you consider your Economic Status? ",best_feature)
    
    #avere qualche statistica
    get_statistica_set(Y_train)
    # Data pre-processing
    from sklearn import preprocessing
    print("\nscaler dataset...")
    scaler = preprocessing.StandardScaler().fit(X_train)
    Xtrain_scaled = scaler.transform(X_train)
    Xval_scaled = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)
    print("done scaled")
        
    
    
    print("\nbuild ora uno split per grid-search")
    #build the split for grid-search
    
    #print(len(Xtrain_scaled))
    #print("____________________")
    #print(len(Xval_scaled))
    
    X_train_val_scaled=np.concatenate((Xtrain_scaled,Xval_scaled), axis=0)
    Y_train_val=np.concatenate((Y_train,Y_val), axis=0)
    #print("ecco concatenato")
    #print(len(X_train_val_scaled))
    #print(len(Y_train_val))
    print("done")
    
    #============ Dummy ============

    algorithm = "Dummy"
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(Xtrain_scaled,Y_train)
    y_dummy = dummy_clf.predict(X_test)
    score = f1_score(Y_test, y_dummy, average="macro")
    print("\nDummy model f1 score",score)
    
    #============ Logistic Regression =============
    algorithm = "LR"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Logistica Regression",score)
    
    #============ Decision Tree =============
    algorithm = "DT"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Decision Tree",score)
    
    #============ Ridged classificer =============
    algorithm = "RI"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Ridged classificer",score)
    
    
    
    
    #con smote 
    
    print("\nCON SMOTE ")
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, Y_train)
    #print(X_resampled)
    #print(y_resampled)
    get_statistica_set(y_resampled)
    
    print("\nscaler smote dataset ...")
    scaler = preprocessing.StandardScaler().fit(X_resampled)
    Xtrain_scaled = scaler.transform(X_resampled)
    Xval_scaled = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)
    print("done scaled smote")
    print("\nbuild ora uno split per grid-search")
    
    
    X_train_val_scaled=np.concatenate((Xtrain_scaled,Xval_scaled), axis=0)
    Y_train_val=np.concatenate((y_resampled,Y_val), axis=0)
    #print("ecco concatenato")
    #print(len(X_train_val_scaled))
    #print(len(Y_train_val))
    print("done")
    
    #============ Dummy ============

    algorithm = "Dummy"
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(Xtrain_scaled,y_resampled)
    y_dummy = dummy_clf.predict(X_test)
    score = f1_score(Y_test, y_dummy, average="macro")
    print("\nDummy model f1 score",score)
    
    #============ Logistic Regression =============
    algorithm = "LR"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Logistica Regression",score)
    
    #============ Decision Tree =============
    algorithm = "DT"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Decision Tree",score)
    
    #============ Ridged classificer =============
    algorithm = "RI"
    clf, param_grid = get_model(algorithm)

    search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=8, refit=True, n_jobs = 8)
    search.fit(X_train_val_scaled, Y_train_val)
    y_test_pred = search.predict(Xtest_scaled)
    score = f1_score(Y_test, y_test_pred, average = 'macro')
    print("Ridged classificer",score)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    split_index = [-1] * len(X_train_df_filtered) + [0] * len(X_val_df_filtered)
    ps = PredefinedSplit(test_fold = split_index)
    # X_train_val = pd.concat([X_train_df_filtered, X_val_df_filtered])
    X_train_val = np.vstack((X_train_df_filtered, X_val_df_filtered))
    y_train_val = y_train.tolist() + y_val.tolist()
    '''
    
    
    
  
    ''' 
               #####3====================================
    NITER = 5

    for iter in range(NITER):
        #sample the list of users
        random.shuffle(males)
        random.shuffle(females)

                #stratified gender split
                tr_users = males[:int(.7 * len(males))] + females[:int(.7 * len(females))]
                val_users = males[int(.7 * len(males)):int(.8 * len(males))] + females[int(.7 * len(females)):int(.8 * len(females))]
                test_users = males[int(.8 * len(males)):] + females[int(.8 * len(females)):]

                #generate the training, val and testing set
                train_df = df[df['user'].isin(tr_users)]
                val_df = df[df['user'].isin(val_users)]
                test_df = df[df['user'].isin(test_users)]
                print(f"Split:\t\tTrain:{len(train_df)}\tVal:{len(val_df)}\tTest:{len(test_df)}")

                #map id_for_ts into the ground truth
                #derive X and y
                y_train =  train_df["user"]
                y_val =  val_df["user"]
                y_test =  test_df["user"]
                y_train = y_train.apply(lambda x: mapper[x])
                y_val = y_val.apply(lambda x: mapper[x])
                y_test = y_test.apply(lambda x: mapper[x])

                X_train = train_df.drop(columns=["user"])
                X_val = val_df.drop(columns=["user"])
                X_test = test_df.drop(columns=["user"])

                #filter the data with tfresh
                X_train_df_filtered = select_features(X_train, y_train, n_jobs = 10)
                X_val_df_filtered = X_val[X_train_df_filtered.columns]
                X_test_df_filtered = X_test[X_train_df_filtered.columns]
                print(f"\n\nNew shape: {X_train_df_filtered.shape}")

                columns_before_str = X_train_df_filtered.columns

                #add features micro/macro task
                X_train_df_filtered["macro_task"] = [x.split("|")[0] for x in X_train_df_filtered.index]
                X_train_df_filtered["micro_task"] = [x.split("|")[-1][:4] for x in X_train_df_filtered.index]
                X_val_df_filtered["macro_task"] = [x.split("|")[0] for x in X_val_df_filtered.index]
                X_val_df_filtered["micro_task"] = [x.split("|")[-1][:4] for x in X_val_df_filtered.index]
                X_test_df_filtered["macro_task"] = [x.split("|")[0] for x in X_test_df_filtered.index]
                X_test_df_filtered["micro_task"] = [x.split("|")[-1][:4] for x in X_test_df_filtered.index]

                X_train_df_filtered = pd.get_dummies(X_train_df_filtered, columns=['macro_task', 'micro_task'], drop_first=True)
                X_val_df_filtered = pd.get_dummies(X_val_df_filtered, columns=['macro_task', 'micro_task'], drop_first=True)
                X_test_df_filtered = pd.get_dummies(X_test_df_filtered, columns=['macro_task', 'micro_task'], drop_first=True)


                #scaler
                ct = ColumnTransformer([('scaler', StandardScaler(), columns_before_str)], remainder='passthrough')
                scl = ct.fit(X_train_df_filtered)#.to_numpy())
                X_train_df_filtered = scl.transform(X_train_df_filtered)#.to_numpy())
                X_val_df_filtered = scl.transform(X_val_df_filtered)#.to_numpy())
                X_test_df_filtered = scl.transform(X_test_df_filtered)#.to_numpy())


                #build the split for grid-search
                split_index = [-1] * len(X_train_df_filtered) + [0] * len(X_val_df_filtered)
                ps = PredefinedSplit(test_fold = split_index)
                # X_train_val = pd.concat([X_train_df_filtered, X_val_df_filtered])
                X_train_val = np.vstack((X_train_df_filtered, X_val_df_filtered))
                y_train_val = y_train.tolist() + y_val.tolist()

                #============ Dummy ============

                algorithm = "Dummy"
                dummy_clf = DummyClassifier(strategy="stratified")
                dummy_clf.fit(X_train,y_train)
                y_dummy = dummy_clf.predict(X_test)
                score = f1_score(y_test, y_dummy, average="macro")

                res_df.loc[len(res_df)] = [iter, algorithm, t, s, score]

                #============ Logistic Regression =============
                algorithm = "LR"
                clf, param_grid = get_model(algorithm)

                search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=ps, refit=True, n_jobs = 8)
                search.fit(X_train_val, y_train_val)
                y_test_pred = search.predict(X_test_df_filtered)
                score = f1_score(y_test, y_test_pred, average = 'macro')

                res_df.loc[len(res_df)] = [iter, algorithm, t, s, score]

                res_df.to_csv(f"ablation_log_VR_gender.csv")
'''