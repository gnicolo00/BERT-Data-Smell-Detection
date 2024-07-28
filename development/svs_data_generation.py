from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)
synthetic_data = synthesizer.sample(num_rows=20000)


def split_address(address):
    pattern = r'^(.*)\n(.*),\s*(\w{2})\s*(\d{5})$'
    match = re.match(pattern, address)
    if match:
        street = match.group(1)
        city = match.group(2)
        state = match.group(3)
        zip_code = match.group(4)
        return street, city, state, zip_code, 'USA'  # Supponiamo che il paese sia sempre USA
    return np.nan, np.nan, np.nan, np.nan, np.nan


address_components = synthetic_data['billing_address'].apply(split_address)
synthetic_data[['street', 'city', 'state', 'zip_code', 'country']] = pd.DataFrame(address_components.tolist(),
                                                                             index=synthetic_data.index)
half_rows = len(synthetic_data) // 2
indices = np.random.permutation(synthetic_data.index)  # Selezione di metà delle righe in maniera casuale
first_half = indices[:half_rows]
second_half = indices[half_rows:]

# La prima metà avrà billing_address riempito e le altre colonne relative all'indirizzo vuote
synthetic_data.loc[first_half, ['street', 'city', 'state', 'zip_code', 'country']] = np.nan
# La seconda metà avrà billing_address vuoto e le altre colonne relative all'indirizzo riempite
synthetic_data.loc[second_half, 'billing_address'] = np.nan
# Aggiunta della colonna 'smell' per indicare la presenza o l'assenza di uno Splitted Value Smell
synthetic_data['smell'] = synthetic_data['billing_address'].isna()


pd.set_option('display.max_columns', 7)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

print("Dataset head:")
print(synthetic_data.head(n=20))

print("Address column head:")
columns = ['billing_address', 'street', 'city', 'state', 'zip_code', 'country', 'smell']
print(synthetic_data[columns].head(20))

print("Missing values:")
null_mask = synthetic_data.isna()
null_count = null_mask.sum()
print(null_count)

# Eliminazione delle righe in cui tutti i valori riguardanti gli indirizzi sono mancanti
columns_to_check = ['billing_address', 'street', 'city', 'state', 'zip_code', 'country']
synthetic_data.dropna(subset=columns_to_check, how='all', inplace=True)

# Creazione e salvataggio di un pie plot per illustrare il bilanciamento della classe
fig, ax = plt.subplots(figsize=(7, 6))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
plt.pie(synthetic_data['smell'].value_counts(), labels=['No Smell', 'Smell'], colors=["#E3E3E3", "#28A745"],
        explode=(0, 0.015), autopct="%0.2f", startangle=90, textprops={'fontsize': 11})
plt.savefig(os.path.join("..", "plots", "svs-balancing.png"), format="png")


synthetic_data.to_csv(os.path.join("..", "datasets", "svs_dataset.csv"), index=False)
