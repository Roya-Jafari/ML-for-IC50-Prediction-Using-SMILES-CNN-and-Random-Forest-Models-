import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from rdkit.Chem.Descriptors3D import (
    Asphericity,
    Eccentricity,
    InertialShapeFactor,
    NPR1,
    NPR2,
    RadiusOfGyration
)
# Load CSV
df = pd.read_csv("chembl_activity_1.csv", sep=';', low_memory=False)
ic50 = df[df['Standard Type'] == 'IC50']
ic50_5col = ic50[['Molecule ChEMBL ID', 'Molecular Weight', 'Smiles', 'Standard Type', 'Value']]

# Convert 'Value' to numeric and clean
ic50_5col['Value'] = pd.to_numeric(ic50_5col['Value'], errors='coerce')

# Drop invalid, missing, or zero/negative IC50 values
ic50_5col = ic50_5col.dropna(subset=['Value'])
ic50_5col = ic50_5col[ic50_5col['Value'] > 0]

print(ic50_5col)
# Drop missing SMILES
ic50_5col = ic50_5col.dropna(subset=['Smiles'])
print(ic50_5col)

# Ensure SMILES are strings
ic50_5col['Smiles'] = ic50_5col['Smiles'].astype(str)
print(ic50_5col)

# Remove invalid SMILES using RDKit
from rdkit import Chem
ic50_5col = ic50_5col[
    ic50_5col['Smiles'].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )
].reset_index(drop=True)

print(ic50_5col)
print(ic50_5col.shape)

ic50_5col['pIC50'] = -np.log10(ic50_5col['Value'] * 1e-6)
print(ic50_5col['pIC50'])

# Drop rows with missing SMILES
ic50_5col = ic50_5col.dropna(subset=['Smiles'])

# Create RDKit molecule objects
ic50_5col['Mol'] = ic50_5col['Smiles'].apply(Chem.MolFromSmiles)
print(ic50_5col['Mol'])

# Filter out failed molecule creations
ic50_5col = ic50_5col[ic50_5col['Mol'].notna()].reset_index(drop=True)
print(ic50_5col.shape)
