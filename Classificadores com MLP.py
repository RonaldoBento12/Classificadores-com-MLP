import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


#dados CIFAR-10
print("Carregando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Nomes das classes
class_names = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo', 
               'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

print(f"Dados de treino: {x_train.shape}")
print(f"Dados de teste: {x_test.shape}")
print(f"Total de dados utilizados: {x_train.shape[0] + x_test.shape[0]} imagens")


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)


y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Forma após achatar: {x_train_flat.shape}")
print(f"Forma das labels: {y_train_cat.shape}")


unique, counts = np.unique(y_train, return_counts=True)
print(f"\nDistribuição das classes no dataset de treino:")
for i, (classe, count) in enumerate(zip(class_names, counts)):
    print(f"  {classe}: {count} imagens ({count/len(y_train)*100:.1f}%)")
print(f"Total confirmado: {sum(counts)} imagens de treino")


def create_model(architecture_name, hidden_layers, activation='relu'):
    model = keras.Sequential(name=architecture_name)
    model.add(layers.Input(shape=(3072,)))
    
    
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation=activation))
        model.add(layers.Dropout(0.2))
    
    
    model.add(layers.Dense(10, activation='softmax'))
    
    return model


models_config = {
    'Modelo_1_Simples': {
        'hidden_layers': [128],
        'activation': 'relu',
        'description': '1 camada escondida com 128 neurônios, ativação ReLU'
    },
    'Modelo_2_Medio': {
        'hidden_layers': [256, 128],
        'activation': 'relu',
        'description': '2 camadas escondidas (256, 128), ativação ReLU'
    },
    'Modelo_3_Profundo': {
        'hidden_layers': [512, 256, 128],
        'activation': 'relu',
        'description': '3 camadas escondidas (512, 256, 128), ativação ReLU'
    },
    'Modelo_4_Tanh': {
        'hidden_layers': [256, 128],
        'activation': 'tanh',
        'description': '2 camadas escondidas (256, 128), ativação Tanh'
    },
    'Modelo_5_Largo': {
        'hidden_layers': [1024, 512],
        'activation': 'relu',
        'description': '2 camadas escondidas largas (1024, 512), ativação ReLU'
    }
}


models = {}
histories = {}
predictions = {}
metrics_results = {}

# Treinar cada modelo
for model_name, config in models_config.items():
    
    print(f"TREINANDO {model_name}")
    print(f"Descrição: {config['description']}")

    
    
    model = create_model(
        model_name, 
        config['hidden_layers'], 
        config['activation']
    )
    
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    print(f"\nArquitetura do {model_name}:")
    model.summary()
    
    
    print(f"\nIniciando treinamento do {model_name} com TODOS os 50.000 dados de treino...")
    print(f"Batch size: 128 (irá processar {50000//128 + 1} batches por época)")
    
    history = model.fit(
        x_train_flat, y_train_cat,
        epochs=15, 
        batch_size=128,
        validation_data=(x_test_flat, y_test_cat),
        verbose=1,
        shuffle=True
    )
    
    
    models[model_name] = model
    histories[model_name] = history
    
   
    print(f"\nFazendo predições com {model_name} em TODOS os 10.000 dados de teste...")
    y_pred_prob = model.predict(x_test_flat, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    predictions[model_name] = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    
  
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics_results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\nMétricas do {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


plt.figure(figsize=(20, 12))


plt.subplot(2, 3, 1)
for model_name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{model_name} (treino)')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} (val)', linestyle='--')
plt.title('Accuracy durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)


plt.subplot(2, 3, 2)
for model_name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{model_name} (treino)')
    plt.plot(history.history['val_loss'], label=f'{model_name} (val)', linestyle='--')
plt.title('Loss durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Comparação de métricas
plt.subplot(2, 3, 3)
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('Comparação de Métricas entre Modelos')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#matrizes de confusão
n_models = len(models)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_name, pred_data) in enumerate(predictions.items()):
    if idx < len(axes):
        
        cm = confusion_matrix(pred_data['y_true'], pred_data['y_pred'])
        
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx])
        axes[idx].set_title(f'Matriz de Confusão - {model_name}')
        axes[idx].set_xlabel('Predição')
        axes[idx].set_ylabel('Verdadeiro')


if len(models) < len(axes):
    axes[-1].remove()

plt.tight_layout()
plt.show()

#melhor modelo
best_model_name = max(metrics_results.keys(), 
                     key=lambda x: metrics_results[x]['Accuracy'])


print(f"MELHOR MODELO: {best_model_name}\n")


print(f"\nDescrição: {models_config[best_model_name]['description']}")
print(f"Accuracy: {metrics_results[best_model_name]['Accuracy']:.4f}")

print(f"\nRelatório de Classificação Detalhado:")
print(classification_report(
    predictions[best_model_name]['y_true'],
    predictions[best_model_name]['y_pred'],
    target_names=class_names
))




print("RESUMO COMPARATIVO DE TODOS OS MODELOS\n")


summary_df = pd.DataFrame(metrics_results).T
summary_df = summary_df.round(4)


descriptions = []
for model_name in summary_df.index:
    descriptions.append(models_config[model_name]['description'])

summary_df['Descrição'] = descriptions


summary_df = summary_df[['Descrição', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

print(summary_df.to_string())


print(f"Total de modelos treinados: {len(models)}")
print(f"Melhor modelo: {best_model_name}")
print(f"Melhor accuracy: {metrics_results[best_model_name]['Accuracy']:.4f}")