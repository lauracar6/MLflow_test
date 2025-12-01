import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import json
from sklearn import metrics

df = pd.read_csv('dataset_join_preprocess.csv')

# como as decision trees nao sao sensiveis à magnitude dos dados nao os vou normalizar

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) 

seeds = [1234, 5678, 2025, 2004, 2016]

def random_forest_model(df, seeds):
    # começamos por extrair o X e o y do dataset
    X = df.drop(columns=['Class'])
    y = df['Class']

    dictionaries = [] # criar lista vazia para os dicionários todos
    confusion_matrices = []
    
    # aplicamos aqui o modelo
    for seed in seeds:
     
        rf_model = RandomForestClassifier(random_state=seed, class_weight='balanced')

        rf_model.fit(X, y)

        y_pred = rf_model.predict(X)

        conf_matrix = metrics.confusion_matrix (y,y_pred)
        confusion_matrices.append(conf_matrix)
        #print(f"Matriz de confusão: {conf_matrix}")

        
        #Não sei se estas métricas serão necessárias ter aqui calculadas uma vez que tenho de perceber se o autologging faz isto
        acc = metrics.accuracy_score(y, y_pred=y_pred)
        sens = metrics.recall_score(y, y_pred=y_pred)
        spe = metrics.recall_score(y,y_pred=y_pred,pos_label=0)
        f1 = metrics.f1_score(y, y_pred=y_pred)

        dictionaries.append({
            "Seed": seed,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spe,
            "F1 score": f1
        })

    # guardar resultados num ficheiro json (codigo do geeks for geeks)
    with open('random_forest_results.json', 'w') as json_file: 
        json.dump(dictionaries, json_file, indent=4)
        

    return confusion_matrices, dictionaries
    
rf_conf_matrices, metrics_list = random_forest_model(df,seeds)

print(rf_conf_matrices[1])



    



