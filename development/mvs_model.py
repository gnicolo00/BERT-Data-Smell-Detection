import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings. filterwarnings('ignore')
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from official.nlp import optimization
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

tf.get_logger().setLevel('ERROR')


def data_preprocessing(rows):
    # Ottenimento del meta-dataset contenente gli Splitted Value Smells
    mvs_meta_dataset = pd.read_csv(os.path.join("..", "datasets", "meta-datasets", "mvs_metadataset.csv"))
    mvs_meta_dataset.drop("ID Dataset", axis=1, inplace=True)
    # Unione delle colonne 'Nome Feature' e 'Valore' in un'unica stringa "nome colonna: valore"
    mvs_meta_dataset["Nome e Valore"] = mvs_meta_dataset["Nome Feature"] + ": " + mvs_meta_dataset["Valore"]
    mvs_meta_dataset.drop("Nome Feature", axis=1, inplace=True)
    mvs_meta_dataset.drop("Valore", axis=1, inplace=True)

    # Undersampling dei dati
    smelly = mvs_meta_dataset[mvs_meta_dataset['Smell'] == 1]
    non_smelly = mvs_meta_dataset[mvs_meta_dataset['Smell'] == 0]
    # Selezione casuale del numero di righe passato come parametro per ognuno dei due gruppi di dati (smelly e non)
    smelly_sample = smelly.sample(n=rows, random_state=42)  # random_state per riproducibilità
    non_smelly_sample = non_smelly.sample(n=rows, random_state=42)
    # Unione dei due gruppi in un singolo dataframe
    sampled_dataset = pd.concat([smelly_sample, non_smelly_sample])
    # Shuffle per mescolare le righe casualmente
    sampled_dataset = sampled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled_dataset


def build_classifier_model():
    bert_model_link = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
    bert_preprocess_link = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_link, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model_link, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

    return tf.keras.Model(text_input, net)


def create_classifier_model(df, epochs):
    tensorflow_ds = tf.data.Dataset.from_tensor_slices(dict(df))

    # Selezione dei parametri della rete neurale
    steps_per_epoch = tf.data.experimental.cardinality(tensorflow_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    # Definizione dell'optimizer, della loss e delle metriche da utilizzare
    optimizer = optimization.create_optimizer(init_lr=3e-5,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.metrics.BinaryAccuracy(), tf.metrics.Precision(), tf.metrics.Recall()]
    # Creazione, compilazione e training del modello
    classifier_model = build_classifier_model()
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    return classifier_model


def train_and_evaluate(epochs, X_train, X_test, y_train, y_test):
    # Training del modello
    history = classifier_model.fit(x=X_train,
                                   y=y_train,
                                   validation_data=(X_test, y_test),
                                   epochs=epochs)

    classifier_model.save("mvs_model", include_optimizer=False)

    # Salvataggio delle performance del modello
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    prec = history_dict['precision']
    val_prec = history_dict['val_precision']
    rec = history_dict['recall']
    val_rec = history_dict['val_recall']

    print("[Training] Multiple Value Smell Model Metrics:\n"
          f"Accuracy: {acc}\n"
          f"Precision: {prec}\n"
          f"Recall: {rec}\n")
    print("[Testing] Multiple Value Smell Model Metrics:\n"
          f"Accuracy: {val_acc}\n"
          f"Precision: {val_prec}\n"
          f"Recall: {val_rec}\n")

    save_evaluation_graphs(acc, val_acc, loss, val_loss, X_test, y_test)


def save_evaluation_graphs(acc, val_acc, loss, val_loss, X_test, y_test):
    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss - Multiple Value Smells Model')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy - Multiple Value Smells Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join("..", "plots", "mvs-acc-and-loss.png"))
    plt.show()


    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=5.0)
    # Confusion matrix
    y_pred_prob = classifier_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    color_map = ListedColormap('white', name='colormap_list')
    color_matrix = [['#FFFFFF', '#28A745'], ['#28A745', '#FFFFFF']]
    color_text_matrix = [['black', 'white'], ['white', 'black']]
    plt.imshow(cm, cmap=color_map, origin='upper')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), color=color_text_matrix[i][j])
            plt.fill_between([j - 0.5, j + 0.5], i - 0.5, i + 0.5, color=color_matrix[i][j], alpha=1)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join("..", "plots", "mvs-confusion-matrix.png"))
    plt.show()


if __name__ == "__main__":
    df = data_preprocessing(250)

    # Ottenimento delle labels
    y = df["Smell"]
    # Ottenimento del dataset senza labels
    X = df.drop("Smell", axis=1)

    # Data splitting (80-20) casuale per ottenere il dataset di training e il dataset di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    epochs = 3

    # Creazione e compilazione del modello
    classifier_model = create_classifier_model(df, epochs)
    # Training del modello
    train_and_evaluate(epochs, X_train, X_test, y_train, y_test)
