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
X = df.drop(columns=['Class'])
y = df['Class']

# como as decision trees nao sao sensiveis à magnitude dos dados nao os vou normalizar

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) 


seeds = [1234, 5678, 2025, 2004, 2016]
dictionaries = [] # criar lista vazia para os dicionários todos

for i in range(len(seeds)):
    seed=seeds[i]

    rf_model = RandomForestClassifier(random_state=seed, class_weight='balanced')

    rf_model.fit(X, y)

    y_pred = rf_model.predict(X)

    #conf_matrix = metrics.confusion_matrix (y,y_pred)
    conf_matrix_param = metrics.confusion_matrix (y,y_pred)
    #print(f"Matriz de confusão: {conf_matrix}")

    # visualizar a matriz de confusão
    class_names=['Vaginal delivery','C-section'] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(conf_matrix_param), annot=True, cmap="YlGnBu" ,fmt='g') #alterar aqui a confusão de matriz que queremos
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Matriz de confusão (lr best parameters) - SEED {seed}', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()

    # Calculo e visualização de métricas:
    acc = metrics.accuracy_score(y, y_pred=y_pred)
    sens = metrics.recall_score(y, y_pred=y_pred)
    spe = metrics.recall_score(y,y_pred=y_pred,pos_label=0)
    f1 = metrics.f1_score(y, y_pred=y_pred)

    print(f"\n--------RUN {i+1} - SEED {seed} - Performance Metrics from Logistic Regression--------")
    print(f"\nAccuracy: {acc}")
    print(f"\nSensitivity: {sens}")
    print(f"\nSpecificity: {spe}")
    print(f"\nF1 score: {f1}")

    # guardar resultados num ficheiro json (codigo do geeks for geeks)

    dictionary = {
        "Seed": seed,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spe,
        "F1 score": f1
    }

    print(f"dictionary for seed {seed}: {dictionary}")

    dictionaries.append(dictionary)
    print(f"dictionaries: {dictionaries}")

    with open('random_forest_results.json', 'w') as json_file: 
        json.dump(dictionaries, json_file, indent=4)

print(f"dictionaries: {dictionaries}")

