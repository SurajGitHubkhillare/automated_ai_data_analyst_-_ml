import pandas as pd
import numpy as np

np.random.seed(42)

n = 100
data = {
    "case_id": [f"EZYV{str(i).zfill(4)}" for i in range(1, n+1)],
    "continent": np.random.choice(["Asia", "Europe", "North America", "South America", "Africa", "Oceania"], n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
    "education_of_employee": np.random.choice(["High School", "Master's", "Bachelor's", "Doctorate"], n, p=[0.1, 0.3, 0.4, 0.2]),
    "has_job_experience": np.random.choice(["Y", "N"], n, p=[0.7, 0.3]),
    "requires_job_training": np.random.choice(["Y", "N"], n, p=[0.2, 0.8]),
    "no_of_employees": np.random.randint(10, 50000, n),
    "yr_of_estab": np.random.randint(1800, 2023, n),
    "region_of_employment": np.random.choice(["Northeast", "South", "Midwest", "West", "Island"], n, p=[0.25, 0.3, 0.2, 0.2, 0.05]),
    "prevailing_wage": np.round(np.random.uniform(20000, 150000, n), 2),
    "unit_of_wage": np.random.choice(["Hour", "Year", "Week", "Month"], n, p=[0.1, 0.8, 0.05, 0.05]),
    "full_time_position": np.random.choice(["Y", "N"], n, p=[0.8, 0.2]),
    "case_status": np.random.choice(["Certified", "Denied"], n, p=[0.65, 0.35])
}

df = pd.DataFrame(data)
df.to_csv("visa_dataset_100_rows.csv", index=False)
print("Created visa_dataset_100_rows.csv successfully.")
