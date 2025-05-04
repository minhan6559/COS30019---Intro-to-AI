"""
Script to combine original cleaned data with predictions
Creates complete October-November datasets for each model
"""

import pandas as pd
import os
from datetime import datetime

# Configuration
CLEANED_DATA_PATH = '/content/drive/MyDrive/processed_data/cleaned_data.csv'
PREDICTIONS_BASE_PATH = '/content/drive/MyDrive/predictions/'
OUTPUT_BASE_PATH = '/content/drive/MyDrive/complete_predictions/'

# Create output directory
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# Load original cleaned data
print("Loading original cleaned data...")
df_clean = pd.read_csv(CLEANED_DATA_PATH)
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Filter for October and November 2006
oct_nov_start = pd.Timestamp('2006-10-01')
oct_nov_end = pd.Timestamp('2006-11-30 23:59:59')
df_oct_nov = df_clean[
    (df_clean['Date'] >= oct_nov_start) & 
    (df_clean['Date'] <= oct_nov_end)
].copy()

print(f"Original October-November data: {len(df_oct_nov)} records")

# Get list of model folders
model_folders = [f for f in os.listdir(PREDICTIONS_BASE_PATH) 
                 if os.path.isdir(os.path.join(PREDICTIONS_BASE_PATH, f))]

print(f"\nFound models: {model_folders}")

for model_name in model_folders:
    print(f"\nProcessing model: {model_name}")
    
    # Load predictions for this model
    pred_file = os.path.join(PREDICTIONS_BASE_PATH, model_name, 'traffic_predictions_oct_nov_2006.csv')
    
    if not os.path.exists(pred_file):
        print(f"  Prediction file not found: {pred_file}")
        continue
    
    df_pred = pd.read_csv(pred_file)
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    
    print(f"  Loaded {len(df_pred)} predictions")
    
    # Create a copy of October-November data for this model
    df_combined = df_oct_nov.copy()
    
    # Add a column to mark original vs predicted data
    df_combined['data_source'] = 'original'
    df_pred['data_source'] = 'predicted'
    
    # Rename predicted traffic column to match original
    df_pred = df_pred.rename(columns={'predicted_traffic': 'traffic_volume'})
    
    # Add missing columns to predictions (copy from original data where possible)
    # Get a sample row to understand structure
    sample_row = df_oct_nov.iloc[0]
    
    # Add missing columns to predictions
    for col in df_oct_nov.columns:
        if col not in df_pred.columns:
            if col in ['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']:
                # These are location-specific, need to map from Location
                location_mapping = df_oct_nov.groupby('Location').first()[col].to_dict()
                df_pred[col] = df_pred['Location'].map(location_mapping)
            elif col not in ['traffic_volume', 'Location', 'Date', 'interval_id', 'time_of_day']:
                # For other columns, set to None or default value
                df_pred[col] = None
    
    # Ensure columns are in the same order
    df_pred = df_pred[df_combined.columns]
    
    # Combine original and predictions
    df_combined = pd.concat([df_combined, df_pred], ignore_index=True)
    
    # Sort by Location, Date, and interval_id
    df_combined = df_combined.sort_values(['Location', 'Date', 'interval_id'])
    
    # Save combined data
    output_file = os.path.join(OUTPUT_BASE_PATH, f'{model_name}_complete_oct_nov_2006.csv')
    df_combined.to_csv(output_file, index=False)
    
    print(f"  Saved combined data to: {output_file}")
    print(f"  Total records: {len(df_combined)}")
    print(f"  Original records: {len(df_combined[df_combined['data_source'] == 'original'])}")
    print(f"  Predicted records: {len(df_combined[df_combined['data_source'] == 'predicted'])}")
    
    # Create a summary
    summary = {
        'total_records': len(df_combined),
        'original_records': len(df_combined[df_combined['data_source'] == 'original']),
        'predicted_records': len(df_combined[df_combined['data_source'] == 'predicted']),
        'locations': df_combined['Location'].nunique(),
        'date_range': f"{df_combined['Date'].min()} to {df_combined['Date'].max()}"
    }
    
    # Save summary
    import pickle
    summary_file = os.path.join(OUTPUT_BASE_PATH, f'{model_name}_summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)

print("\nAll models processed successfully!")
print(f"Combined data saved to: {OUTPUT_BASE_PATH}")

# Create a comparison view
print("\nCreating comparison summary...")
comparison_data = []

for model_name in model_folders:
    combined_file = os.path.join(OUTPUT_BASE_PATH, f'{model_name}_complete_oct_nov_2006.csv')
    if os.path.exists(combined_file):
        df = pd.read_csv(combined_file)
        comparison_data.append({
            'Model': model_name,
            'Total Records': len(df),
            'Original': len(df[df['data_source'] == 'original']),
            'Predicted': len(df[df['data_source'] == 'predicted']),
            'Locations': df['Location'].nunique()
        })

df_comparison = pd.DataFrame(comparison_data)
comparison_file = os.path.join(OUTPUT_BASE_PATH, 'model_comparison_summary.csv')
df_comparison.to_csv(comparison_file, index=False)
print(f"Comparison summary saved to: {comparison_file}")
print("\nComparison Summary:")
print(df_comparison)
