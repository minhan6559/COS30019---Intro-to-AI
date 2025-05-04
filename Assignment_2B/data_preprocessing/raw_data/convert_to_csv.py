import pandas as pd

# New file path for the uploaded .xlsx file
xlsx_path = "Assignment_2B/Scats Data October 2006.xlsx"

# Attempt to read the .xlsx file using openpyxl engine
excel_data = pd.read_excel(xlsx_path, engine="openpyxl")

# Save it as a CSV file
csv_path = "Assignment_2B/Scats Data October 2006.csv"
# excel_data.to_csv(csv_path, index=False)

# Read the CSV file back
csv_data = pd.read_csv(csv_path)
print(csv_data.head())
