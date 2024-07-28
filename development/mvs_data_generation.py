from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer
from faker import Faker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

real_data, metadata = download_demo(modality='single_table', dataset_name='student_placements')

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)
synthetic_data = synthesizer.sample(num_rows=20000)

fake = Faker()


def generate_personal_info(gender):
    if gender == 'M':
        first_name = fake.first_name_male()
    else:
        first_name = fake.first_name_female()
    last_name = fake.last_name()
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=25).strftime("%Y-%m-%d")

    return f"{first_name} {last_name}, {birth_date}, {gender}"


def generate_names(gender):
    if gender == 'M':
        first_name = fake.first_name_male()
    else:
        first_name = fake.first_name_female()
    last_name = fake.last_name()

    return f"{first_name} {last_name}"


def detect_smell(personal_info):
    return len(personal_info.split(',')) == 3  # Tre elementi: 1) nome e cognome 2) data di nascita 3) sesso


half_rows = len(synthetic_data) // 2
indices = np.random.permutation(synthetic_data.index)  # Selezione di metà delle righe in maniera casuale
first_half = indices[:half_rows]
second_half = indices[half_rows:]

# La prima metà avrà la colonna personal_info riempita con i dati anagrafici (nome e cognome, data di nascita e sesso)
synthetic_data.loc[first_half, 'personal_info'] = synthetic_data.loc[first_half, 'gender'].apply(generate_personal_info)
# La seconda metà avrà la colonna personal_info riempita solamente con nome e cognome
synthetic_data.loc[second_half, 'personal_info'] = synthetic_data.loc[second_half, 'gender'].apply(generate_names)
# Aggiunta della colonna 'smell' per indicare la presenza o l'assenza di un Multiple Value Smell
synthetic_data['smell'] = synthetic_data['personal_info'].apply(detect_smell)

# Spostamento della colonna personal_info in seconda posizione
synthetic_data.insert(1, 'personal_info', synthetic_data.pop('personal_info'))
synthetic_data.drop(columns=['gender'], inplace=True)


pd.set_option('display.max_columns', 7)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

print("Dataset head:")
print(synthetic_data.head(n=20))

print("Personal info column head:")
columns = ['personal_info', 'smell']
print(synthetic_data[columns].head(20))

print("Missing values: ", end='')
null_mask = synthetic_data['personal_info'].isna()
null_count = null_mask.sum()
print(null_count)

# Creazione e salvataggio di un pie plot per illustrare il bilanciamento della classe
fig, ax = plt.subplots(figsize=(7, 6))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
plt.pie(synthetic_data['smell'].value_counts(), labels=['No Smell', 'Smell'], colors=["#E3E3E3", "#28A745"],
        explode=(0, 0.015), autopct="%0.2f", startangle=90, textprops={'fontsize': 11})
plt.savefig(os.path.join("..", "plots", "mvs-balancing.png"), format="png")


synthetic_data.to_csv(os.path.join("..", "datasets", "mvs_dataset.csv"), index=False)
