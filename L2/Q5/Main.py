import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
pop_file = os.path.join(script_dir, 'TAB2818_sv.csv')
death_file = os.path.join(script_dir, 'TAB959_sv.csv')

# 1. Load the data using comma separation
df_pop = pd.read_csv(pop_file, sep=',', encoding='latin1')
df_death = pd.read_csv(death_file, sep=',', encoding='latin1')

def process_scb_data(df, value_col):
    # Filter for the years 2021 to 2025
    df_filtered = df[df['år'].isin([2021, 2022, 2023, 2024, 2025])].copy()
    
    # Extract just the integer from the 'ålder' (age) column using a raw string r'(\d+)'
    extracted_age = df_filtered['ålder'].astype(str).str.extract(r'(\d+)')
    df_filtered['age'] = extracted_age[0]
    
    # Drop rows where no age was found (e.g., "Totalt" summary rows)
    df_filtered = df_filtered.dropna(subset=['age']).copy()
    
    # Now it is safe to convert to integer
    df_filtered['age'] = df_filtered['age'].astype(int)
    
    # Group by age and gender (kön), then calculate the mean across the 5 years
    avg_data = df_filtered.groupby(['age', 'kön'])[value_col].mean().reset_index()
    
    # Pivot the table to have ages as rows and genders as columns
    pivot_data = avg_data.pivot(index='age', columns='kön', values=value_col)
    
    # Translate columns for the plot
    if 'kvinnor' in pivot_data.columns and 'män' in pivot_data.columns:
        pivot_data = pivot_data.rename(columns={'kvinnor': 'Women', 'män': 'Men'})
    
    # Change the legend title from "kön" to "Gender"
    pivot_data.columns.name = 'Gender'
    
    return pivot_data


# 2. Process the data
avg_pop = process_scb_data(df_pop, 'Medelfolkmängd (efter födelseår)')
avg_death = process_scb_data(df_death, 'Döda, antal')

# 3. Create the plots for Task 5a and 5b
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 5a: Average Population
avg_pop.plot(ax=ax1, color=['blue', 'red'])
ax1.set_title('5a: Average Mean Population by Age (2021-2025)')
ax1.set_xlabel('Age')
ax1.set_ylabel('Population')
ax1.grid(True, alpha=0.3)

# Plot 5b: Average Deaths
avg_death.plot(ax=ax2, color=['blue', 'red'])
ax2.set_title('5b: Average Number of Deaths by Age (2021-2025)')
ax2.set_xlabel('Age')
ax2.set_ylabel('Number of Deaths')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()