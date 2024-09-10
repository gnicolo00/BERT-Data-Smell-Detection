import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# Raccolta di tutti i dataset nella cartella "sub-datasets/svs"
csv_files = glob.glob(os.path.join("..", "datasets",  "sub-datasets", "svs", "*.csv"))
datasets = []
for file in csv_files:
    df = pd.read_csv(file)
    datasets.append(df)


# Trasformazione di un singolo dataset in meta-dataset mantenendo l'ordine delle righe
def transform_to_meta(dataset, id_dataset):
    sub_meta_dataset_rows = []
    num_columns = len(dataset.columns)

    for index in range(len(dataset)):
        for i, column in enumerate(dataset.columns):
            if i == num_columns - 1: # Tutte le colonne di ogni dataset sono smelly, eccetto l'ultima
                smell_value = 0
            else:
                smell_value = 1

            sub_meta_dataset_rows.append({
                'ID Dataset': id_dataset,
                'Nome Feature': column,
                'Valore': dataset.at[index, column],
                'Smell': smell_value
            })

    return pd.DataFrame(sub_meta_dataset_rows)


sub_meta_datasets = []
for i, dataset in enumerate(datasets):
    sub_meta_datasets.append(transform_to_meta(dataset, i+1))

# Unione dei meta-dataset in uno solo
svs_meta_dataset = pd.concat(sub_meta_datasets, ignore_index=True)
svs_meta_dataset.to_csv(os.path.join("..", "datasets",  "meta-datasets", "svs_metadataset.csv"), index=False)


# Visualizzazione del numero dei valori mancanti del meta-dataset
print("Meta-dataset Missing Values:")
null_mask = svs_meta_dataset.isna()
null_count = null_mask.sum()
print(null_count)

# Creazione e salvataggio di un pie plot
fig, ax = plt.subplots(figsize=(7, 6))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
plt.pie(svs_meta_dataset['Smell'].value_counts(), labels=['Smelly', 'Non-Smelly'], colors=['#28A745', '#CCCCCC'],
        explode=(0, 0.015), autopct='%0.2f', startangle=90, textprops={'fontsize': 11})
plt.savefig(os.path.join("..", "plots", "svs-balancing-pie.png"), format="png")

# Creazione e salvataggio di un bar plot
smell_counts = svs_meta_dataset['Smell'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(smell_counts.index, smell_counts.values, color=['#28A745', '#CCCCCC'], edgecolor='black', linewidth=1)
plt.title('Distribuzione dei dati Smelly vs Non-Smelly', fontsize=16)
plt.xlabel('Splitted Value Smell', fontsize=14)
plt.ylabel('Numero di Istanze', fontsize=14)
plt.xticks([0, 1], ['Non-Smelly', 'Smelly'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join("..", "plots", "svs-balancing-bar.png"), format="png")
