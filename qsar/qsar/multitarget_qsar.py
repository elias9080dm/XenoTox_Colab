import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
import io
from pathlib import Path
import base64
import warnings
from qsar.descriptors import calcular_descriptores

RDLogger.DisableLog("rdApp.*")

# ------------------------------------------------------------------
# Clase MultiTargetQSAR
# ------------------------------------------------------------------
class MultiTargetQSAR:
    """
    Herramienta QSAR multiblanco para predicción binaria
    de activación de receptores moleculares.
    """

    def __init__(self, model_paths: dict):
        """
        Parameters
        ----------
        model_paths : dict
            Diccionario con el nombre del blanco y la ruta al .pkl

            Ejemplo:
            {
                "ahr": "models/best_model_ahr_xgb_optuna.pkl",
                "car": "models/best_model_car_xgb_optuna.pkl",
                "pxr": "models/best_model_pxr_xgb_optuna.pkl"
            }
        """
        self.models = {}

        for target, path in model_paths.items():
            obj = joblib.load(path)

            # Validación estricta del contenido del pkl
            required_keys = ["model", "label_encoder", "X_train_preprocessed"]
            for key in required_keys:
                if key not in obj:
                    raise ValueError(
                        f"El modelo '{target}' no contiene la clave obligatoria '{key}'"
                    )

            self.models[target] = obj

    # ------------------------------------------------------------------
    # Estandarización química
    # ------------------------------------------------------------------
    def _standardize_molecule(self, mol):
        """
        Aplica estandarización química consistente:
        - Fragmento mayor
        - Normalización
        - Reionización
        - Neutralización
        """
        lfc = rdMolStandardize.LargestFragmentChooser()
        normalizer = rdMolStandardize.Normalizer()
        reionizer = rdMolStandardize.Reionizer()
        uncharger = rdMolStandardize.Uncharger()

        try:
            mol = lfc.choose(mol)
            mol = normalizer.normalize(mol)
            mol = reionizer.reionize(mol)
            mol = uncharger.uncharge(mol)
            return mol
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Imagen molecular (base64)
    # ------------------------------------------------------------------
    def _mol_to_base64(self, mol, size=(200, 200)):
        if mol is None:
            return None
        try:
            img = Draw.MolToImage(mol, size=size)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dominio de aplicabilidad (Leverage)
    # ------------------------------------------------------------------
    def _compute_leverage(self, X_ext, X_train):
        """
        Calcula leverage y AD flag usando el criterio del Williams plot
        """
        try:
            X_train_df = pd.DataFrame(X_train)
            n, p = X_train_df.shape
            h_star = 3 * (p + 1) / n

            H = X_ext @ np.linalg.pinv(X_train_df.T @ X_train_df) @ X_ext.T
            hat = np.diag(H)

            ad_flag = [
                "Inside AD" if h < h_star else "Outside AD"
                for h in hat
            ]

            return hat, ad_flag

        except Exception:
            return [np.nan] * len(X_ext), ["AD Error"] * len(X_ext)

    # ------------------------------------------------------------------
    # Normalizar input
    # ------------------------------------------------------------------
    def _load_input_data(self,input_data):
        """
        Acepta:
        - lista de SMILES
        - ruta a archivo CSV
        - DataFrame
        
        Devuelve:
        - DataFrame con columna 'SMILES'
        """

        # Caso 1: lista de SMILES
        if isinstance(input_data, list):
            df = pd.DataFrame({"SMILES": input_data})

        # Caso 2: ruta a CSV
        elif isinstance(input_data, (str, Path)):
            input_data = Path(input_data)
            if not input_data.exists():
                raise FileNotFoundError(f"No se encontró el archivo: {input_data}")
            
            df = pd.read_csv(input_data)

            if "SMILES" not in df.columns:
                raise ValueError("El archivo CSV debe contener una columna llamada 'SMILES'")

        # Caso 3: DataFrame
        elif isinstance(input_data, pd.DataFrame):
            if "SMILES" not in input_data.columns:
                raise ValueError("El DataFrame debe contener una columna 'SMILES'")
            df = input_data.copy()

        else:
            raise TypeError(
                "input_data debe ser una lista de SMILES, un DataFrame o la ruta a un CSV"
            )

        # Limpieza básica
        df = df.dropna(subset=["SMILES"])
        df["SMILES"] = df["SMILES"].astype(str)

        return df.reset_index(drop=True)


    # ------------------------------------------------------------------
    # Validación de smiles de entrada
    # ------------------------------------------------------------------
    def _validate_input_smiles(self, df):
        """
        Validación previa del input de SMILES.
        Espera un DataFrame con columna 'SMILES'.
        Emite warnings si hay SMILES inválidos.
        """

        if "SMILES" not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'SMILES'")

        invalid_rows = []
        valid_rows = []

        for idx, smi in enumerate(df["SMILES"]):

            # No string
            if not isinstance(smi, str):
                invalid_rows.append(idx)
                continue

            # Heurística: posible concatenación accidental
            if len(smi) > 300:
                invalid_rows.append(idx)
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_rows.append(idx)
            else:
                valid_rows.append(idx)

        df_valid = df.loc[valid_rows].reset_index(drop=True)
        df_invalid = df.loc[invalid_rows].reset_index(drop=True)

        report = {
            "input_smiles": len(df),
            "valid_smiles": len(df_valid),
            "invalid_smiles": len(df_invalid)
        }

        if report["invalid_smiles"] > 0:
            warnings.warn(
                f"Input SMILES: {report['input_smiles']} | "
                f"Válidos: {report['valid_smiles']} | "
                f"Inválidos: {report['invalid_smiles']}",
                UserWarning
            )

        return df_valid, df_invalid, report


    # ------------------------------------------------------------------
    # Predicción principal
    # ------------------------------------------------------------------
    def predict(self, input_data):

      # Normalización de input
      df = self._load_input_data(input_data)

      # Validación SOLO para warning
      df_valid, df_invalid, report = self._validate_input_smiles(df)

      results = []

      # Iterar sobre la columna SMILES
      for smiles in df["SMILES"]:

          row = {
              "SMILES": smiles,
              "Valid_SMILES": True
          }

          # ---------- Parseo SMILES ----------
          mol = Chem.MolFromSmiles(smiles)
          if mol is None:
              row["Valid_SMILES"] = False
              results.append(row) 
              continue

          mol = self._standardize_molecule(mol)
          if mol is None:
              row["Valid_SMILES"] = False
              results.append(row)
              continue

          smiles_std = Chem.MolToSmiles(mol)
          row["Molecule_Image"] = self._mol_to_base64(mol)

          # ---------- Descriptores ----------
          descriptors = calcular_descriptores(smiles_std)
          if descriptors is None:
              row["Valid_SMILES"] = False
              results.append(row)
              continue

          X_raw = np.asarray(descriptors, dtype=float).reshape(1, -1)

          # ---------- Predicción por target ----------
          for target, obj in self.models.items():
              clf = obj["model"]
              le = obj["label_encoder"]
              X_train = obj["X_train_preprocessed"]

              Xp = clf.named_steps["clean"].transform(X_raw)
              Xp = clf.named_steps["imputer"].transform(Xp)
              Xp = clf.named_steps["scaler"].transform(Xp)

              class_index = list(le.classes_).index("Active")
              proba_active = clf.predict_proba(Xp)[0, class_index]

              pred_enc = clf.predict(Xp)[0]
              pred_label = le.inverse_transform([pred_enc])[0]

              leverage, ad_flag = self._compute_leverage(Xp, X_train)

              row[f"{target}_Pred"] = pred_label
              row[f"{target}_Prob"] = proba_active
              row[f"{target}_Leverage"] = leverage[0]
              row[f"{target}_AD"] = ad_flag[0]

          results.append(row)

      final_df = pd.DataFrame(results)

      return final_df
