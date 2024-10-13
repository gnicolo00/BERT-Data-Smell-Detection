import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as text

@st.cache_resource()
def load_models():
    svs_model = tf.keras.models.load_model('svs_model')
    mvs_model = tf.keras.models.load_model('mvs_model')
    return svs_model, mvs_model


def print_my_examples(inputs, results):
    # Sort in base agli score assegnati
    sorted_data = sorted(
        [(inputs[i], results[i][0]) for i in range(len(inputs)) if results[i][0] >= 0.5],
        key=lambda x: x[1],
        reverse=True # Ordine decrescente
    )

    # Formattazione dei risultati per la visualizzazione
    result_for_printing = [f'{input_val:<30} : {result_val:.6f}' for input_val, result_val in sorted_data]

    st.session_state['output'] = result_for_printing


def predict_smell(model, dataframe, column_name):
    inputs = []
    for i in dataframe[column_name]:
        inputs.append(i)

    results = tf.sigmoid(model(tf.constant(inputs)))

    print_my_examples(inputs, results)

if __name__ == "__main__":

    if 'output' not in st.session_state:
        st.session_state['output'] = ""

    st.title(":orange[BERT] :gray[Data Smell Detection]")

    svs_model, mvs_model = load_models()

    selected_model_string = st.radio("Which model do you want to use?",
                                    ["Splitted Value Smells Detection", "Multiple Value Smells Detection"],
                                    horizontal=True)

    if selected_model_string == "Splitted Value Smells Detection":
        selected_model = svs_model
    else:
        selected_model = mvs_model

    dataset = st.file_uploader("Carica il dataset", type={"csv"})

    if dataset is not None:
        dataset_df = pd.read_csv(dataset)
        st.write(dataset_df)
        dataframe_columns = dataset_df.columns

        col1, col2, col3, col4 = st.columns(4, gap='small')
        i = 0
        for column in dataframe_columns:
            if i % 4 == 0:
                with col1:
                    if st.button(column):
                        predict_smell(selected_model, dataset_df, column)
            elif i % 4 == 1:
                with col2:
                    if st.button(column):
                        predict_smell(selected_model, dataset_df, column)
            elif i % 4 == 2:
                with col3:
                    if st.button(column):
                        predict_smell(selected_model, dataset_df, column)
            elif i % 4 == 3:
                with col4:
                    if st.button(column):
                        predict_smell(selected_model, dataset_df, column)
            i = i + 1

    st.write(st.session_state['output'])