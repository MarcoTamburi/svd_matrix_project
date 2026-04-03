DESCRIZIONE (ITALIANO)


Questo software permette l’analisi di dati di Circular Dichroism (CD) combinando:
Decomposizione SVD (Singular Value Decomposition)
Modelli termodinamici (unfolding a 2, 3 o 4 stati)
Ricostruzione spettrale e validazione del modello
L’obiettivo è fornire una pipeline modulare, riproducibile e user-friendly per analisi spettroscopiche avanzate.

WORKFLOW
Il workflow è diviso in tre notebook principali.
-01_n_components.ipynb
Preparazione dati
Costruzione matrice spettrale
SVD
Scelta numero componenti
Salvataggio:
U_prime.csv
V_prime.csv
session_config.json
Questo step va eseguito quando cambiano i dati o il numero di componenti.
-02_run_fit.ipynb
Esecuzione fit termodinamico
Salvataggio risultati in:
results/fit3_run/fit3_<timestamp>/
Contiene:
parametri finali
configurazione usata
output del fit
Questo step genera una nuova run completa e riproducibile.
-03_spectral_reconstruction.ipynb
Carica automaticamente l’ultima run disponibile
Ricostruisce:
spettri puri
popolazioni
spettri completi
Confronta dati sperimentali vs modello
Calcola metriche di errore
Questo step NON riesegue il fit.
Serve per analisi, validazione e visualizzazione.

STRUTTURA DEL PROGETTO
src/ → motore del software
notebooks/ → interfaccia utente
configs/ → configurazioni statiche
data/ → dati e output intermedi
results/ → risultati delle run

NOTE IMPORTANTI
Ogni run è salvata separatamente (riproducibilità completa)
Il notebook di reconstruction non rilancia il fit
I dati sono separati da configurazioni e risultati
Tutti i risultati sono tracciabili tramite cartelle timestampate

------------------------------------------------------------------------

DESCRIPTION (ENGLISH)


This software performs Circular Dichroism (CD) data analysis by combining:
Singular Value Decomposition (SVD)
Thermodynamic unfolding models (2-state / 3-state)
Spectral reconstruction and validation
The goal is to provide a modular, reproducible and user-friendly pipeline.

WORKFLOW
The workflow is divided into three main notebooks.
-01_n_components.ipynb
Data preparation
Spectral matrix construction
SVD decomposition
Component selection
Saves:
U_prime.csv
V_prime.csv
session_config.json
Run this step when data or number of components changes.
-02_run_fit.ipynb
Runs thermodynamic fitting
Saves results in:
results/fit3_run/fit3_<timestamp>/
Includes:
fitted parameters
config snapshot
fit outputs
Each execution creates a new reproducible run.
03_spectral_reconstruction.ipynb
Automatically loads latest run
Reconstructs:
state spectra
populations
full spectra
Compares experimental vs model
Computes error metrics
This step does NOT rerun the fit.

PROJECT STRUCTURE
src/ → core engine
notebooks/ → user interface
configs/ → static configs
data/ → processed data
results/ → fit outputs

KEY FEATURES
Fully reproducible runs
Decoupled fitting and reconstruction
Modular architecture
Suitable for publication-level analysis
