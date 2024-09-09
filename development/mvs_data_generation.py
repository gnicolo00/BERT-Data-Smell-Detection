import pandas as pd
import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from faker import Faker


# Rimozione delle colonne non necessarie e delle righe con valori nulli o duplicati
def delete_col_and_rows(df, cols_to_save):
    df = df[cols_to_save]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=cols_to_save, inplace=True)
    return df


# Creazione e addestramento di un synthesizer per dataset e generazione di dati sintetici
def get_synthetic_data(df, cols, function, new_column):
    df_metadata = SingleTableMetadata()
    df_metadata.detect_from_dataframe(df)
    df_metadata.primary_key = None
    df_synthesizer = GaussianCopulaSynthesizer(df_metadata)
    df_synthesizer.fit(df)
    df_synthetic_data = df_synthesizer.sample(num_rows=5000)

    # Conversione di tutte le colonne in stringa, poiché BERT lavora solo con stringhe
    df_synthetic_data = df_synthetic_data.astype(str)

    # Applicazione della funzione di merge
    merged_column = df_synthetic_data[cols].apply(function, axis=1)
    df_synthetic_data[new_column] = merged_column

    return df_synthetic_data


def merge_columns(row):
    return ','.join(row.values)


def generate_personal_info(num_rows):
    data = []

    for _ in range(num_rows):
        gender = faker.random_element(elements=("M", "F"))
        if gender == "M":
            name = faker.name_male()
        else:
            name = faker.name_female()
        date_of_birth = faker.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d")

        data.append({
            "name": name,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "personal_info": f"{name},{date_of_birth},{gender}"
        })

    return pd.DataFrame(data)


def generate_contact_info(num_rows):
    data = []

    for _ in range(num_rows):
        email = faker.email()
        phone_number = faker.phone_number()

        data.append({
            "email": email,
            "phone_number": phone_number,
            "contact_info": f"{email},{phone_number}"
        })

    return pd.DataFrame(data)


def generate_company_hierarchy(num_rows):
    data = []

    for _ in range(num_rows):
        ceo_name = faker.name()
        cto_name = faker.name()
        cfo_name = faker.name()

        data.append({
            "ceo_name": ceo_name,
            "cto_name": cto_name,
            "cfo_name": cfo_name,
            "company_hierarchy": f"{ceo_name},{cto_name},{cfo_name}"
        })

    return pd.DataFrame(data)


def generate_weather(num_rows):
    data = []

    for _ in range(num_rows):
        temperature = faker.random_number(digits=2)
        humidity = faker.random_number(digits=2)
        pressure = faker.random_number(digits=4)

        data.append({
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "weather": f"{temperature},{humidity},{pressure}"
        })

    return pd.DataFrame(data)


def generate_trip_details(num_rows):
    data = []

    for _ in range(num_rows):
        departure_time = faker.date_time_this_year().strftime("%Y-%m-%d %H:%M:%S")
        arrival_time = faker.date_time_this_year().strftime("%Y-%m-%d %H:%M:%S")
        duration = faker.random_number(digits=3)

        data.append({
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "duration": duration,
            "trip_details": f"{departure_time},{arrival_time},{duration}"
        })

    return pd.DataFrame(data)


def generate_flight_details(num_rows):
    data = []

    for _ in range(num_rows):
        flight_number = faker.bothify(text='??###')
        departure = faker.city()
        destination = faker.city()

        data.append({
            "flight_number": flight_number,
            "departure": departure,
            "destination": destination,
            "flight_details": f"{flight_number},{departure},{destination}"
        })

    return pd.DataFrame(data)


cars_details_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "cars_details.csv"))
cars_specs_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "cars_specs.csv"))
credit_cards_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "credit_cards.csv"))
demographics_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "demographics.csv"))
employees_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "employees.csv"))
medical_records_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "medical_records.csv"))
orders_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "orders.csv"))
passports_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "passports.csv"))
product_info_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "product_info.csv"))
product_specs_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "product_specs.csv"))
projects_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "projects.csv"))
shows_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "shows.csv"))
smartphones_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "smartphones.csv"))
transactions_ds = pd.read_csv(os.path.join("..", "datasets",  "sub-datasets", "mvs", "transactions.csv"))
faker = Faker()

# Dettagli dell'auto
cols = ["year", "manufacturer", "model"]
cars_details_ds = delete_col_and_rows(cars_details_ds, cols)
synthetic_cars_details_ds = get_synthetic_data(cars_details_ds, cols, merge_columns, "car_details")
synthetic_cars_details_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "cars_details.csv"), index=False)
# Specifiche dell'auto
cols = ["condition", "cylinders", "fuel"]
cars_specs_ds = delete_col_and_rows(cars_specs_ds, cols)
synthetic_cars_specs_ds = get_synthetic_data(cars_specs_ds, cols, merge_columns, "car_specs")
synthetic_cars_specs_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "cars_specs.csv"), index=False)
# Carte di credito
cols = ["credit_card", "country_code", "card_type"]
credit_cards_ds = delete_col_and_rows(credit_cards_ds, cols)
synthetic_credit_cards_ds = get_synthetic_data(credit_cards_ds, cols, merge_columns, "credit_card_info")
synthetic_credit_cards_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "credit_cards.csv"), index=False)
# Dati demografici
cols = ["job", "marital", "education"]
demographics_ds = delete_col_and_rows(demographics_ds, cols)
synthetic_demographics_ds = get_synthetic_data(demographics_ds, cols, merge_columns, "demographic_info")
synthetic_demographics_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "demographics.csv"), index=False)
# Dipendenti
cols = ["job_title", "department_name", "length_of_service"]
employees_ds = delete_col_and_rows(employees_ds, cols)
synthetic_employees_ds = get_synthetic_data(employees_ds, cols, merge_columns, "employee_info")
synthetic_employees_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "employees.csv"), index=False)
# Cartelle cliniche
cols = ["medical_condition", "medication", "test_results"]
medical_records_ds = delete_col_and_rows(medical_records_ds, cols)
synthetic_medical_records_ds = get_synthetic_data(medical_records_ds, cols, merge_columns, "medical_record")
synthetic_medical_records_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "medical_records.csv"),
                                    index=False)
# Ordini
cols = ["order_number", "order_date", "product_price"]
orders_ds = delete_col_and_rows(orders_ds, cols)
synthetic_orders_ds = get_synthetic_data(orders_ds, cols, merge_columns, "order_details")
synthetic_orders_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "orders.csv"), index=False)
# Passaporti
cols = ["country", "rank", "year"]
passports_ds = delete_col_and_rows(passports_ds, cols)
synthetic_passports_ds = get_synthetic_data(passports_ds, cols, merge_columns, "passport_info")
synthetic_passports_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "passports.csv"), index=False)
# Info dei prodotti
cols = ["brand", "product_name", "category"]
product_info_ds = delete_col_and_rows(product_info_ds, cols)
synthetic_product_info_ds = get_synthetic_data(product_info_ds, cols, merge_columns, "product_info")
synthetic_product_info_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "product_info.csv"), index=False)
# Specifiche dei prodotti
cols = ["size", "color", "price"]
product_specs_ds = delete_col_and_rows(product_specs_ds, cols)
synthetic_product_info_ds = get_synthetic_data(product_specs_ds, cols, merge_columns, "product_specs")
synthetic_product_info_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "product_specs.csv"),
                                 index=False)
# Progetti
cols = ["repo_name", "username", "repo_url"]
projects_ds = delete_col_and_rows(projects_ds, cols)
synthetic_projects_ds = get_synthetic_data(projects_ds, cols, merge_columns, "project_details")
synthetic_projects_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "projects.csv"), index=False)
# Film e serie TV
cols = ["title", "type", "country"]
shows_ds = delete_col_and_rows(shows_ds, cols)
synthetic_shows_ds = get_synthetic_data(shows_ds, cols, merge_columns, "show_info")
synthetic_shows_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "shows.csv"), index=False)
# Smartphones
cols = ["brand_name", "model_name", "os"]
smartphones_ds = delete_col_and_rows(smartphones_ds, cols)
synthetic_smartphones_ds = get_synthetic_data(smartphones_ds, cols, merge_columns, "phone_info")
synthetic_smartphones_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "smartphones.csv"), index=False)
# Transazioni
cols = ["transaction_amount", "merchant_name", "date"]
transactions_ds = delete_col_and_rows(transactions_ds, cols)
synthetic_transactions_ds = get_synthetic_data(transactions_ds, cols, merge_columns, "transaction_details")
synthetic_transactions_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "transactions.csv"), index=False)
# Dati anagrafici
personal_info_ds = generate_personal_info(5000)
personal_info_ds = personal_info_ds.astype(str)
personal_info_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "personal_info.csv"), index=False)
# Informazioni di contatto
contacts_info_ds = generate_contact_info(5000)
contacts_info_ds = contacts_info_ds.astype(str)
contacts_info_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "contacts.csv"), index=False)
# Gerarchia aziendale
company_hierarchy_ds = generate_company_hierarchy(5000)
company_hierarchy_ds = company_hierarchy_ds.astype(str)
company_hierarchy_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "company_hierarchy.csv"), index=False)
# Dati del meteo
weather_ds = generate_weather(5000)
weather_ds = weather_ds.astype(str)
weather_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "weather.csv"), index=False)
# Viaggi
trips_ds = generate_trip_details(5000)
trips_ds = trips_ds.astype(str)
trips_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "trips.csv"), index=False)
# Voli
flights_ds = generate_flight_details(5000)
flights_ds = flights_ds.astype(str)
flights_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "mvs", "flights.csv"), index=False)
