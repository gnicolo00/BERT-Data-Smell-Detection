import pandas as pd
import os
import re
from urllib.parse import urlparse
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from faker import Faker


# Rimozione delle colonne non necessarie e delle righe con valori nulli o duplicati
def delete_col_and_rows(df, col_to_save):
    df.drop(columns=[col for col in df.columns if col != col_to_save], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=[col_to_save], inplace=True)
    return df


# Creazione e addestramento di un synthesizer per dataset e generazione di dati sintetici
def get_synthetic_data(df, col, function, new_columns):
    df_metadata = SingleTableMetadata()
    df_metadata.detect_from_dataframe(df)
    df_metadata.primary_key = None
    df_synthesizer = GaussianCopulaSynthesizer(df_metadata)
    df_synthesizer.fit(df)
    df_synthetic_data = df_synthesizer.sample(num_rows=5000)

    # Conversione di tutte le colonne in stringa, poiché BERT lavora solo con stringhe
    df_synthetic_data = df_synthetic_data.astype(str)

    # Applicazione della funzione di split
    splitted_components = df_synthetic_data[col].apply(function)
    df_synthetic_data[new_columns] = pd.DataFrame(splitted_components.tolist(), index=df_synthetic_data.index)
    df_synthetic_data[col] = df_synthetic_data.pop(col) # Riordinamento delle colonne
    df_synthetic_data.dropna(inplace=True) # Rimozione delle righe con valori nulli creati dalla funzione di split

    return df_synthetic_data


def split_address(address):
    address = address.strip()
    match = re.match(r"(.+?),\s*(.+?),\s*([A-Z]{2})\s*(\d{4,5})", address)

    if match:
        street = match.group(1)
        city = match.group(2)
        state = match.group(3)
        zip_code = match.group(4)
        return street, city, state, zip_code
    else:
        return None, None, None, None


def split_isbn(isbn):
    if len(isbn) == 13:
        prefix = isbn[:3]
        registration_group = isbn[3]
        publisher_code = isbn[4:7]
        item_number = isbn[7:12]
        check_digit = isbn[12]
        return prefix, registration_group, publisher_code, item_number, check_digit
    else:
        return None, None, None, None, None


def split_license_plate(license_plate):
    match = re.match(r"([A-Z]{3})(\d{2,3}[A-Z]?)", license_plate)

    if match:
        plate_letters = match.group(1)
        plate_numbers = match.group(2)
        return plate_letters, plate_numbers
    else:
        return None, None


def split_location(location):
    parts = location.split(',')

    if len(parts) == 2:
        city = parts[0].strip()
        state = parts[1].strip()
        return city, state
    else:
        return None, None


def split_phone(phone):
    phone = phone.strip().replace('(', '').replace(')', '').replace('-', ' ')
    parts = phone.split()

    if len(parts) == 3:
        area_code = parts[0]
        exchange_code = parts[1]
        line_number = parts[2]
        return area_code, exchange_code, line_number
    else:
        return None, None, None


def split_url(url):
    parsed_url = urlparse(url)
    protocol = parsed_url.scheme

    domain = parsed_url.netloc
    tld = domain.split('.')[-1] if '.' in domain else ''

    return protocol, domain, tld


def generate_birth_dates(num_rows):
    data = []

    for _ in range(num_rows):
        birth_date = faker.date_of_birth()

        data.append({
            'birth_day': birth_date.day,
            'birth_month': birth_date.month,
            'birth_year': birth_date.year,
            'birth_date': birth_date
        })

    return pd.DataFrame(data)


def generate_dates(num_rows):
    data = []

    for _ in range(num_rows):
        date = faker.date()

        data.append({
            'day': pd.to_datetime(date).day,
            'month': pd.to_datetime(date).month,
            'year': pd.to_datetime(date).year,
            'date': date
        })

    return pd.DataFrame(data)


def generate_time(num_rows):
    data = []

    for _ in range(num_rows):
        time = faker.time()
        h, m, s = time.split(':')

        data.append({
            'hours': int(h),
            'minutes': int(m),
            'seconds': int(s),
            'time': time
        })

    return pd.DataFrame(data)


def generate_names(num_rows):
    data = []

    for _ in range(num_rows):
        first_name = faker.first_name()
        second_name = faker.first_name()
        last_name = faker.last_name()
        full_name = f"{first_name} {second_name} {last_name}"

        data.append({
            'first_name': first_name,
            'second_name': second_name,
            'last_name': last_name,
            'full_name': full_name
        })

    return pd.DataFrame(data)


def generate_heights(num_rows):
    data = []

    for _ in range(num_rows):
        feet = faker.random_int(min=4, max=7)
        inches = faker.random_int(min=0, max=11)
        height = f"{feet} ft {inches} in"

        data.append({
            'height_feet': feet,
            'height_inches': inches,
            'height': height
        })

    return pd.DataFrame(data)


def generate_heights_with_units(num_rows):
    data = []

    for _ in range(num_rows):
        height_value = faker.random_int(min=1, max=250)
        height_unit = faker.random_element(elements=["mm", "cm", "m"])
        height = f"{height_value}{height_unit}"

        data.append({
            'height_value': height_value,
            'height_unit': height_unit,
            'height': height
        })

    return pd.DataFrame(data)


def generate_weights_with_units(num_rows):
    data = []

    for _ in range(num_rows):
        weight = faker.random_int(min=40, max=150)  # Peso in kg
        unit = faker.random_element(elements=["kg", "lb"])
        weight_str = f"{weight}{unit}"

        data.append({
            'weight': weight,
            'weight_unit': unit,
            'weight_combined': weight_str
        })

    return pd.DataFrame(data)


def generate_prices(num_rows):
    data = []

    for _ in range(num_rows):
        price_value = faker.random_int(min=1, max=1000)
        currency = faker.currency_code()
        price_str = f"{price_value} {currency}"

        data.append({
            'price_value': price_value,
            'price_currency': currency,
            'price': price_str
        })

    return pd.DataFrame(data)


def generate_coordinates(num_rows):
    data = []

    for _ in range(num_rows):
        latitude = faker.latitude()
        longitude = faker.longitude()
        coordinates = f"{latitude}, {longitude}"

        data.append({
            'latitude': latitude,
            'longitude': longitude,
            'coordinates': coordinates
        })

    return pd.DataFrame(data)


def generate_file_names(num_rows):
    data = []

    for _ in range(num_rows):
        file_name = faker.file_name().split('.')[0]
        extension = faker.file_extension()
        full_file_name = f"{file_name}.{extension}"

        data.append({
            'file_name': file_name,
            'file_extension': extension,
            'file_name_with_extension': full_file_name
        })

    return pd.DataFrame(data)


def generate_exam_datetime(num_rows):
    data = []

    for _ in range(num_rows):
        exam_date = faker.date()
        exam_time = faker.time()
        exam_datetime = f"{exam_date} {exam_time}"

        data.append({
            'exam_date': exam_date,
            'exam_time': exam_time,
            'exam_datetime': exam_datetime
        })

    return pd.DataFrame(data)


def generate_rgb_colors(num_rows):
    data = []

    for _ in range(num_rows):
        primary_color = faker.random_int(min=1, max=255)
        secondary_color = faker.random_int(min=1, max=255)
        tertiary_color = faker.random_int(min=1, max=255)
        rgb_color = f"{primary_color}, {secondary_color}, {tertiary_color}"

        data.append({
            'primary_color': primary_color,
            'secondary_color': secondary_color,
            'tertiary_color': tertiary_color,
            'rgb_color': rgb_color
        })

    return pd.DataFrame(data)


def generate_emails(num_rows):
    data = []

    for _ in range(num_rows):
        username = faker.user_name()
        domain = faker.domain_name()
        email = f"{username}@{domain}"

        data.append({
            'username': username,
            'domain': domain,
            'email': email
        })

    return pd.DataFrame(data)


addresses_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "addresses.csv"))
isbn_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "isbn.csv"))
license_plates_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "license_plates.csv"))
locations_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "locations.csv"))
phone_numbers_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "phone_numbers.csv"))
urls_ds = pd.read_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "urls.csv"))
faker = Faker()

# Indirizzi
addresses_ds = delete_col_and_rows(addresses_ds, 'address')
new_columns = ['street', 'city', 'state', 'zip_code']
synthetic_addresses_ds = get_synthetic_data(addresses_ds, 'address', split_address, new_columns)
synthetic_addresses_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "addresses.csv"), index=False)
# ISBN
isbn_ds = delete_col_and_rows(isbn_ds, 'isbn13')
new_columns = ['prefix', 'registration_group', 'publisher_code', 'item_number', 'check_digit']
synthetic_isbn_ds = get_synthetic_data(isbn_ds, 'isbn', split_isbn, new_columns)
synthetic_isbn_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "isbn.csv"), index=False)
# Targhe
license_plates_ds = delete_col_and_rows(license_plates_ds, 'license_plate')
new_columns = ['plate_letters', 'plate_numbers']
synthetic_license_plates_ds = get_synthetic_data(license_plates_ds, 'license_plate', split_license_plate,
                                                 new_columns)
synthetic_license_plates_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "license_plates.csv"),
                                   index=False)
# Località
locations_ds = delete_col_and_rows(locations_ds, 'location')
new_columns = ['city', 'state']
synthetic_locations_ds = get_synthetic_data(locations_ds, 'location', split_location, new_columns)
synthetic_locations_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "locations.csv"), index=False)
# Numeri di telefono
phone_numbers_ds = delete_col_and_rows(phone_numbers_ds, 'phone_number')
new_columns = ['area_code', 'exchange_code', 'line_number']
synthetic_phone_numbers_ds = get_synthetic_data(phone_numbers_ds, 'phone_number', split_phone, new_columns)
synthetic_phone_numbers_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "phone_numbers.csv"),
                                  index=False)
# URLs
urls_ds = delete_col_and_rows(urls_ds, 'url')
new_columns = ['protocol', 'domain', 'tld']
synthetic_urls_ds = get_synthetic_data(urls_ds, 'url', split_url, new_columns)
synthetic_urls_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "urls.csv"), index=False)
# Date di nascita
birth_dates_ds = generate_birth_dates(5000)
birth_dates_ds = birth_dates_ds.astype(str)
birth_dates_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "birth_dates.csv"), index=False)
# Date
dates_ds = generate_dates(5000)
dates_ds = dates_ds.astype(str)
dates_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "dates.csv"), index=False)
# Orari
times_ds = generate_time(5000)
times_ds = times_ds.astype(str)
times_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "time.csv"), index=False)
# Nomi
names_ds = generate_names(5000)
names_ds = names_ds.astype(str)
names_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "names.csv"), index=False)
# Altezze
heights_ds = generate_heights(5000)
heights_ds = heights_ds.astype(str)
heights_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "heights.csv"), index=False)
# Altezze con unità di misura
heights_units_ds = generate_heights_with_units(5000)
heights_units_ds = heights_units_ds.astype(str)
heights_units_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "heights_units.csv"), index=False)
# Pesi con unità di misura
weights_units_ds = generate_weights_with_units(5000)
weights_units_ds = weights_units_ds.astype(str)
weights_units_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "weights_units.csv"), index=False)
# Prezzi
prices_ds = generate_prices(5000)
prices_ds = prices_ds.astype(str)
prices_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "prices.csv"), index=False)
# Coordinate
coordinates_ds = generate_coordinates(5000)
coordinates_ds = coordinates_ds.astype(str)
coordinates_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "coordinates.csv"), index=False)
# Nomi di file
file_names_ds = generate_file_names(5000)
file_names_ds = file_names_ds.astype(str)
file_names_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "file_names.csv"), index=False)
# Date e orari di esami
exam_datetime_ds = generate_exam_datetime(5000)
exam_datetime_ds = exam_datetime_ds.astype(str)
exam_datetime_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "exam_datetime.csv"), index=False)
# Colori RGB
rgb_colors_ds = generate_rgb_colors(5000)
rgb_colors_ds = rgb_colors_ds.astype(str)
rgb_colors_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "rgb_colors.csv"), index=False)
# Indirizzi email
emails_ds = generate_emails(5000)
emails_ds = emails_ds.astype(str)
emails_ds.to_csv(os.path.join("..", "datasets", "sub-datasets", "svs", "emails.csv"), index=False)
