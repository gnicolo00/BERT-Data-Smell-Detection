# BERT Data Smell Detection

Il tool BERT Data Smell Detection è una web app il cui scopo è permettere agli utenti di inserire ed analizzare dataset alla ricerca di Splitted Value Smells e Multiple Value Smells.

## Struttura delle cartelle

Questa repository è organizzata in tre directory principali:
1. `datasets/`: Contiene i dati di input utilizzati nel progetto.
   - `meta-datasets/`: Dataset che aggregano i dati elaborati e generati per l'addestramento del modello.
   - `sub-datasets/`: Dataset più specifici, utilizzati per analisi e test particolari.
2. `development/`: Include tutti gli script di sviluppo e di generazione dati.
   - `interface.py`: Script per l’interfaccia utente.
   - `mvs_data_generation.py` e `svs_data_generation.py`: Script per generare i dataset per Multiple Value Smells e Splitted Value Smells.
   - `mvs_metadataset.py` e `svs_metadataset.py`: Script per la creazione dei meta-dataset per Multiple Value Smells e Splitted Value Smells.
   - `mvs_model.py` e `svs_model.py`: Scripts per la creazione e addestramento dei modelli per l’identificazione dei Multiple e Splitted Value Smells.
3. `plots/`: Contiene i grafici dei risultati dei modelli, incluse metriche come matrice di confusione, accuracy e loss.

## Avvio del Tool

L'applicazione può essere avviata tramite il file `interface.py`, che utilizza Streamlit per creare l'interfaccia utente. Per avviare l'applicazione, seguire i seguenti passaggi:
1. **Installazione delle Dipendenze**: Installare tutte le dipendenze necessarie riportate nel file `requirements.txt`, il che è possibile farlo tramite il comando `pip install -r requirements.txt`.
2. **Addestramento dei Modelli**: Prima dell'avvio, addestrare i due modelli (oppure il singolo modello che si intende utilizzare) eseguendo i corrispettivi script.
2. **Avvio dell'Applicazione**: Una volta installate le dipendenze, eseguire il comando `streamlit run development/interface.py` dalla directory principale del progetto.
3. **Accesso all'interfaccia**: Dopo l'avvio, Streamlit aprirà automaticamente una nuova scheda del browser. Da qui è possibile interagire con l'applicazione per identificare i Multiple Value Smells e gli Splitted Value Smells nei dataset.

## Funzionamento

Dopo aver eseguito i passi precedenti, l'app dovrebbe risultare attiva e funzionante. Qui di seguito è riportato un breve esempio di workflow:
1. Nella schermata iniziale, selezionare il tipo di data smell che desidera analizzare.
2. Caricare un file CSV da sottoporre al modello. Il tool ne visualizzerà un'anteprima.
3. Selezionare la colonna in cui svolgere l'analisi.
4. Il modello calcola la probabilità di ogni valore di essere un data smell, evidenziando quelli con il punteggio più alto, dunque con una più alta probabilità di essere dannosi.
