import pandas as pd
import numpy as np
import random

# Configuration
INPUT_FILE = "upi_fraud_dataset_balanced.csv"
OUTPUT_FILE = "upi_fraud_dataset_balanced.csv" # Overwrites the file. Change name if you want a copy.
ROWS_TO_ADD = 2000  # Change this to add more or fewer rows

def generate_synthetic_data(num_rows, start_id):
    data = []
    
    for i in range(num_rows):
        current_id = start_id + i + 1
        
        # Decide if this row is Fraud (1) or Safe (0)
        # We keep it roughly balanced to match your current dataset style, 
        # or you can set p=[0.9, 0.1] for a realistic imbalanced dataset.
        is_fraud = np.random.choice([0, 1], p=[0.5, 0.5])
        
        # --- LOGIC FOR REALISM ---
        
        # 1. Transaction Hour (0-23)
        # Fraud often happens late at night, safe usually day time.
        if is_fraud:
            # Higher chance of late night (23, 0, 1, 2, 3)
            trans_hour = np.random.choice(list(range(0, 24)), p=[0.08]*5 + [0.0315]*19)
        else:
            # Random distribution, slightly higher in day
            trans_hour = np.random.randint(6, 23) if random.random() > 0.1 else np.random.randint(0, 24)

        # 2. Transaction Amount
        # Fraud can be small testing amounts or large drains.
        if is_fraud:
            # Mix of small and high amounts
            trans_amount = random.choice([random.uniform(1000, 50000), random.uniform(1, 100)])
        else:
            # Mostly normal spending
            trans_amount = random.uniform(10, 5000)

        # 3. Other fields (Randomized within valid ranges)
        trans_day = np.random.randint(1, 31)
        trans_month = np.random.randint(1, 13)
        trans_year = 2023 # Updating year to be current
        category = np.random.randint(0, 15) # Assuming roughly 15 categories
        
        # Generate a fake UPI number (10 digits)
        upi_number = random.randint(7000000000, 9999999999)
        
        age = np.random.randint(18, 90)
        state = np.random.randint(0, 50)
        zip_code = np.random.randint(10000, 99999)

        # Round amount to 2 decimal places
        trans_amount = round(trans_amount, 2)

        data.append([
            current_id, trans_hour, trans_day, trans_month, trans_year,
            category, upi_number, age, trans_amount, state, zip_code, is_fraud
        ])

    return data

# --- EXECUTION ---
try:
    # Load existing data to preserve schema and ID count
    df_existing = pd.read_csv(INPUT_FILE)
    max_id = df_existing['Id'].max()
    print(f"Current dataset has {len(df_existing)} rows. Max ID: {max_id}")

    # Generate new data
    print(f"Generating {ROWS_TO_ADD} new realistic rows...")
    new_data = generate_synthetic_data(ROWS_TO_ADD, max_id)
    
    # Create DataFrame for new data
    columns = ['Id', 'trans_hour', 'trans_day', 'trans_month', 'trans_year', 
               'category', 'upi_number', 'age', 'trans_amount', 'state', 'zip', 'fraud_risk']
    df_new = pd.DataFrame(new_data, columns=columns)

    # Append and Save
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Added {ROWS_TO_ADD} rows. Total rows: {len(df_final)}")
    print(f"Saved to: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"❌ Error: Could not find {INPUT_FILE}. Make sure it is in the same folder.")