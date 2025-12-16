import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics
import os
import seaborn as sns

# imports realted to mlflow
import mlflow
import mlflow.sklearn

# For finding the best model
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

# confirmar que é este script que está a ser executado (do chat)
print("=== SCRIPT RANDOM_FOREST.PY A EXECUTAR ===")


# antes de começar verificar se a working directory corresponde com a diretoria em que o mlflow está a ser trabalhado (do chat)
print("Working directory:", os.getcwd()) #vai me dizer qual é a diretoria do ficheiro que estou a correr
print("MLflow tracking URI:", mlflow.get_tracking_uri()) #diz-me a diretoria onde as runs do mlflow estão a acontecer (estão a ser loaded?)

df = pd.read_csv('dataset_join_preprocess.csv')

# como as decision trees nao sao sensiveis à magnitude dos dados nao os vou normalizar

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) 

seeds = [1234, 5678, 2025, 2004, 2016] # neste caso uma seed diferente vai corresponder a uma run diferente

# começamos por extrair o X e o y do dataset
X = df.drop(columns=['Class'])
y = df['Class']

classes = ['Vaginal Delivery', 'C-section'] # do chat

nome_experiencia = "TesteModeloRF" # It can contain whitespaces or special characters, but it will make code commands harder to perform

# Provide an Experiment description that will appear in the UI
descrição_experiencia = (
"Teste de MLflow com um modelo de Random Forest que treina e testa com todos os dados"
)

experiment_tags = {
    "mlflow.note.content": descrição_experiencia,
}

# Check if the experiment already exists, if not, create it
experiment = mlflow.get_experiment_by_name(nome_experiencia)
if experiment is None:
    mlflow.create_experiment(
    name=nome_experiencia, tags=experiment_tags
    )

# definir a experiencia para esta que acabou de ser criada
mlflow.set_experiment(experiment_name=nome_experiencia)

# aplicamos aqui o modelo
for seed in seeds:

    rf_model = RandomForestClassifier(random_state=seed, class_weight='balanced')

    test_run_name = f"Manual_run_test_seed_{seed}"

    with mlflow.start_run(run_name=test_run_name):

        rf_model.fit(X, y)

        y_pred = rf_model.predict(X)

        conf_matrix = metrics.confusion_matrix (y,y_pred)
        #print(f"Matriz de confusão: {conf_matrix}")

        # plot da matriz de confusão
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)

        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title(f'Matriz de confusão - SEED {seed}', y=1.1)
        plt.ylabel('Label Verdadeiro')
        plt.xlabel('Label Previsto')

        fig_path = f"confusion_matrix_{seed}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path)
        
        #Não sei se estas métricas serão necessárias ter aqui calculadas uma vez que tenho de perceber se o autologging faz isto
        acc = metrics.accuracy_score(y, y_pred=y_pred)
        mlflow.log_metric("accuracy", acc)
        sens = metrics.recall_score(y, y_pred=y_pred)
        mlflow.log_metric("sensitivity", sens)
        spe = metrics.recall_score(y,y_pred=y_pred,pos_label=0)
        mlflow.log_metric("specificity", spe)
        f1 = metrics.f1_score(y, y_pred=y_pred)
        mlflow.log_metric("f1_score", f1)

    #mlflow.end_run() # aparentemente nao usar isto segundo o chat

"""
experiencia_atual = dict(mlflow.get_experiment_by_name(nome_experiencia))

id_experiencia = experiencia_atual["experiment_id"] # Extract the experiment ID

df_busca_resultados = mlflow.search_runs(
    experiment_ids=id_experiencia, # Define the scope of the search
    run_view_type=ViewType.ALL, # Select the type of run
    output_format="pandas", # Select the output format
)

print(df_busca_resultados)

"""  

    



    



