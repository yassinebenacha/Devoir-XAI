import pandas as pd
import json
import re

# Get features
df = pd.read_csv('fifa.csv')
target_col = 'Man of the Match'
# The notebook used this logic:
# numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
# numeric_features = [c for c in numeric_features if c != 'target']
# Note: 'target' is not in the CSV, it's created in the notebook.
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
# If 'target' was saved to csv in previous steps (unlikely), we exclude it.
# Check what features were used in the notebook.
# The notebook creates 'target' column but doesn't save it to CSV.
# So X = df[numeric_features] where numeric_features is from original CSV.
# We just need to exclude 'Man of the Match' if it was numeric? No it's "Yes"/"No".

numeric_features = [c for c in numeric_features]
# Wait, are there any other columns excluded?
# "numeric_features = [c for c in numeric_features if c != 'target']"
# Since 'target' isn't in df yet, this list is just all numeric columns.

# Get SHAP values
try:
    nb = json.load(open('exercise2_shap.ipynb', encoding='utf-8'))
    # Cell 5 is the one with KernelSHAP
    outputs = nb['cells'][5]['outputs']
    stdout = [o['text'] for o in outputs if 'name' in o and o['name']=='stdout']

    if not stdout:
        print("No stdout found in cell 5.")
        # Try finding in all cells
        for i, cell in enumerate(nb['cells']):
             if 'outputs' in cell:
                 o_list = [o['text'] for o in cell['outputs'] if 'name' in o and o['name']=='stdout']
                 for o in o_list:
                     txt = "".join(o)
                     if "Kernel SHAP values" in txt:
                         stdout = [o]
                         print(f"Found in cell {i}")
                         break
    
    if not stdout:
        print("Still no stdout found.")
        exit()

    text = "".join(stdout[0]) 
    # extract content inside [ ... ]
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        values_str = match.group(1)
        values = [float(x) for x in values_str.split()]
        
        with open('shap_results.txt', 'w') as f:
            f.write(f"{'Feature':<25} | {'SHAP Value'}\n")
            f.write("-" * 40 + "\n")
            
            # Combine and sort by absolute value descending
            feature_shap = list(zip(numeric_features, values))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feat, val in feature_shap:
                f.write(f"{feat:<25} | {val:.6f}\n")
            
            f.write(f"\nSum of SHAP: {sum(values)}\n")
            print("Results written to shap_results.txt")
    else:
        print("Could not parse values from text:", text)

except Exception as e:
    print(e)
