import pandas as pd
import glob
import os

# Raccolta di tutti i dataset nella cartella "sub-datasets/mvs"
csv_files = glob.glob(os.path.join("..", "datasets",  "sub-datasets", "mvs", "*.csv"))
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
            if i == num_columns - 1: # L'ultima colonna di ogni dataset è smelly, tutte le altre no
                smell_value = 1
            else:
                smell_value = 0

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
mvs_meta_dataset = pd.concat(sub_meta_datasets, ignore_index=True)
mvs_meta_dataset.to_csv(os.path.join("..", "datasets",  "meta-datasets", "mvs_metadataset.csv"), index=False)
