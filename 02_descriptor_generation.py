# -------------------
# Descriptors
# -------------------

# Take only 30,000 molecules
MAX_MOLS = 200000
ic50_5col_minimal = ic50_5col.sample(n=min(MAX_MOLS, len(ic50_5col)), random_state=42).reset_index(drop=True)
print(ic50_5col_minimal.shape)

# -------------------------------
# Settings
# -------------------------------
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors

N_2D_DESC = 32
MORGAN_BITS = 1024
USE_3D = False  # set True if you have Mordred installed

if USE_3D:
    from mordred import Calculator, descriptors
    mordred_calc = Calculator(descriptors, ignore_3D=False)

def mol_to_descriptors(mol):
    """
    Convert an RDKit molecule to a descriptor vector:
    2D descriptors + Morgan fingerprint + optional 3D descriptors.
    Falls back to zeros if computation fails.
    """
    if mol is None:
        n_3d = 6 if USE_3D else 0
        return np.zeros(N_2D_DESC + MORGAN_BITS + n_3d, dtype=np.float64)
def mol_to_descriptors(mol):
    try:
        # --- 2D descriptors
        desc_2d_array = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FpDensityMorgan1(mol),
            Descriptors.FpDensityMorgan2(mol),
            Descriptors.FpDensityMorgan3(mol),
            Descriptors.Ipc(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.VSA_EState9(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Chi2v(mol),
            Descriptors.Chi3v(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.RingCount(mol),
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.SlogP_VSA3(mol),
            Descriptors.SlogP_VSA4(mol)
        ], dtype=np.float64)

        # --- Morgan fingerprint
        #fp_array = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS), dtype=np.float64)

        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        fp_array = np.array(morgan_gen.GetFingerprint(mol), dtype=np.float32)

        # --- Concatenate all
        desc = np.concatenate([desc_2d_array, fp_array])
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        return desc

    except Exception as e:
        return np.zeros(N_2D_DESC + MORGAN_BITS, dtype=np.float64)
      # --- Loop over molecules ---
# Use the DataFrame that actually has the 'Mol' column

descs = []
for mol in ic50_5col_minimal["Mol"]:
    descs.append(mol_to_descriptors(mol))

X = np.stack(descs)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = ic50_5col_minimal['pIC50'].values


# -----------------------------
# CLEAN DESCRIPTORS FOR RF
# -----------------------------
# Replace NaN and inf
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# Optional: clip extreme values
X = np.clip(X, -1e6, 1e6)

# Ensure float64 for RF
X = X.astype(np.float64)

# -----------------------------
# Split into train/test
# -----------------------------
from sklearn.model_selection import train_test_split

y = y.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Descriptor matrix shape:", X.shape)
print("Label array shape:", y.shape)
# -------------------
# Dataset
# -------------------

from sklearn.model_selection import train_test_split

# X and y are already defined
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
