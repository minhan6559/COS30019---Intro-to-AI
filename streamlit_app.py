import os
import sys
import streamlit as st

# Change to the Assignment_2B directory
os.chdir("./Assignment_2B")

# Now import and run the app
sys.path.insert(0, os.getcwd())

# Import after changing directory so relative imports work
from app import TBRGSApp

# Run the application
if __name__ == "__main__":
    app = TBRGSApp("processed_data/preprocessed_data/sites_metadata.json")
    app.run()
