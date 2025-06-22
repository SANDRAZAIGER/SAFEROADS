import pandas as pd
import json

def convert_excel_to_json():
    # Read the Excel file
    df = pd.read_excel('../data/output.xlsx')
    
    # Print information about the data
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Convert the dataframe to a list of dictionaries
    records = df.to_dict('records')
    
    # Save to JSON file
    with open('../data/accidents.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal records converted: {len(records)}")

if __name__ == "__main__":
    convert_excel_to_json() 