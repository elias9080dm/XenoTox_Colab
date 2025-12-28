from rdkit import Chem
from rdkit.Chem import Descriptors

def calcular_descriptores(smiles):
    """
    Calcula descriptores RDKit a partir de un SMILES.
    Devuelve una lista de valores en el orden de Descriptors._descList.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return [func(mol) for _, func in Descriptors._descList]
    except Exception:
        return None
