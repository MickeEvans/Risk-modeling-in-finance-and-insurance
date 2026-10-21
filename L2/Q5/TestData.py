import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
pop_file = os.path.join(script_dir, 'TAB2818_sv.csv')
death_file = os.path.join(script_dir, 'TAB959_sv.csv')

# SCB files often use latin1 encoding or cp1252, and semicolon separators
try:
    df_pop = pd.read_csv(pop_file, sep=';', encoding='latin1', nrows=5)
    print("Population Columns:", df_pop.columns.tolist())
    print(df_pop.head(2))
    
    print("\n-------------------\n")
    
    df_death = pd.read_csv(death_file, sep=';', encoding='latin1', nrows=5)
    print("Death Columns:", df_death.columns.tolist())
    print(df_death.head(2))
except Exception as e:
    print("Error reading files:", e)