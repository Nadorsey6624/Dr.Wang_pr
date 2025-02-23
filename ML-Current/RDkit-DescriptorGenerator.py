# %%
import pandas as pd
import numpy as np
import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# %%
Seed = r"C:\Users\Dillo\OneDrive\Desktop\LCCCSeed-ML-2-2-25-RemovedCommasFromSolventsList.xlsx"
SeedData = pd.read_excel(Seed)


# %%
## A little pubchempy smiles generation from chemical name example -- Unrelated to our data.

# results = pcp.get_compounds('Triethylamine', 'name')
# print(results)

# for compound in results:
#     print(compound.isomeric_smiles)

# %%
print(type(Seed))

# %%
FullSolvents = pd.DataFrame()
FullSolvents['Removed_Spaces'] = SeedData['Solvents'].str.replace(r'; \s*', ';', regex=True)
FullSolvents['Cleaned_Solvents'] = FullSolvents['Removed_Spaces'].str.replace(r' \s*\(near crit\)', '', regex=True)
FullSolvent_split = FullSolvents['Cleaned_Solvents'].str.split(';', expand=True)

UniqueSolvents = pd.unique(FullSolvent_split.stack())
print(type(UniqueSolvents))



# %%
def get_cid(UniqueSolvents):
    try:
        compound = pcp.get_compounds(UniqueSolvents, 'name')
        if compound:
            return int(compound[0].cid)
        else:
            return None
    except Exception as e:
        return None

def get_smiles(compound_name):
    try:
        compound = pcp.get_compounds(compound_name, 'name')
        if compound:
            return compound[0].isomeric_smiles
        else:
            return None
    except Exception as e:
        return None


# %%
# Property Arrays
UniqueSolventsDataFrame = pd.DataFrame()
UniqueSolvents_series = pd.Series(UniqueSolvents)
#smiles = []
UniqueSolventsDataFrame['Solvents']= UniqueSolvents_series



UniqueSolventsDataFrame['Smiles'] = UniqueSolventsDataFrame['Solvents'].apply(get_smiles)

mol_list = []

for smile in UniqueSolventsDataFrame['Smiles']:
    if smile:  # Skip None or empty strings
        mol = Chem.MolFromSmiles(smile)
        if mol:  # Ensure RDKit didn't return None
            mol_list.append(mol)
        else:
            print(f"Warning: Invalid SMILES string - {smile}")  # Debugging info

# Print first few rows for debugging
print(len(list(UniqueSolventsDataFrame['Smiles'])))
print(UniqueSolventsDataFrame.head())
print(f"Number of valid molecules: {len(mol_list)}")

## Apparently can't generate smiles for deuterated acetone - will need to manually create or find in HSP software.

# %%
UniqueSolventsDataFrame

# %%
# Ensure 'Smiles' is treated as a string column and drop invalid values
data = pd.DataFrame()
data['Solvents'] = UniqueSolventsDataFrame['Solvents'].astype(str)
data['Smiles'] = UniqueSolventsDataFrame['Smiles'].astype(str)  # Convert everything to strings

data = data.dropna(subset=['Smiles'])

# Remove NaN, 'None' strings, and empty values
data = data[~data['Smiles'].isin([None, 'None', '', 'nan', 'NaN'])].dropna(subset=['Smiles'])

#data = data.dropna(subset= ['mol'])

smiles_list = data['Smiles'].values
solvents_list = data['Solvents']


data

# %%
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # If invalid SMILES, return None or some placeholder. Better to filter out invalid rows beforehand.
        return None
    ##These are the descriptors Ethier paper utilized - They cut out the descriptors they decided were repetitive - 212 descriptors available cut to 196 I believe
    custom_descriptors = ["BalabanJ", "BertzCT", "Chi0","Chi1", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
                          "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
                          "EState_VSA10", "ExactMolWt", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount", "HeavyAtomMolWt", "lpc", "Kappa1", "Kappa2", "Kappa3",
                          "LabuteASA", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAlphaticRings",
                          "NumAromaticRings", # Upon review I accidently left this descriptor out - Ethier used it.
                          "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumValenceElectrons",
                          "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA12",
                          "PEOE_VSA14", "RingCount", "SMR_VSA1", "SMR_VSA2", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2",
                          "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA11", "TPSA", "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4",
                          "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "fr_Al_OH", "fr_Al_OH_noTert", "fr_C_O", "fr_C_O_noCOO", "fr_NH0",
                          "fr_aldehyde", "fr_allytic_oxid", "fr_aryl_methyl", "fr_benzene", "fr_bicyclic", "fr_epoxide", "fr_ester", "fr_ether", "fr_ketone", "fr_ketone_Topliss",
                          "fr_methoxy", "fr_nitrile", "fr_nitro", "fr_para_hydroxylation", "fr_sulfone", "fr_unbrch_alkane", "MaxAbsEStateIndex", "MaxAbsPartialCharge",
                          "MaxEStateIndex", "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex", "MinPartialCharge"]
    
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(custom_descriptors)
    desc = descriptor_calculator.CalcDescriptors(mol)
    headers = descriptor_calculator.GetDescriptorNames()  # Retrieve headers
    
    return desc, headers, mol

# %%

# Compute descriptors for all molecules


X_list = []
valid_smiles = []
header = []
mol_list = []


for i, s in enumerate(smiles_list):
    desc, headers, mol = compute_descriptors(s)
    if desc is not None:
        X_list.append(desc)
        valid_smiles.append(s)
        header.append(headers)
        mol_list.append(mol)
        
df_descriptors = pd.DataFrame(X_list, columns=headers)
df_descriptors.insert(0, "Smiles", valid_smiles)
df_descriptors.insert(0, "Solvents", solvents_list)

df_descriptors

# %%


morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)


fingerprints = []  #  Morgan fingerprints as bit vectors

for mol in mol_list:
    if mol:
        fp = morgan_gen.GetFingerprint(mol)
        fp_bits = list(fp.GetOnBits())  # Store only "on" bits
    else:
        fp_bits = []
    
    fp_vector = [1 if i in fp_bits else 0 for i in range(1024)]
    fingerprints.append(fp_vector)



fingerprints_df = pd.DataFrame(fingerprints, columns=[f"FP_bit_{i}" for i in range(1024)])




fingerprints_df



# %%
Final_df = pd.concat([df_descriptors, fingerprints_df], axis=1)

Final_df

# %%
#Final_df.to_excel('SolventDescriptors-2-2-25.xlsx', index=False)

# %%



