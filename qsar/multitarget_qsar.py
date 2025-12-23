import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
import io
import base64

from qsar.descriptors import calcular_descriptores

RDLogger.DisableLog("rdApp.*")


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
                "AHR": "models/best_model_AHR_xgb_optuna.pkl",
                "CAR": "models/best_model_CAR_xgb_optuna.pkl",
                "PXR": "models/best_model_PXR_xgb_optuna.pkl"
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
    # Predicción principal
    # ------------------------------------------------------------------
    def predict(self, smiles_list):
        """
        Realiza predicción multiblanco para una lista de SMILES
        """
        all_results = []
        invalid_smiles = []

        for smiles in smiles_list:
            # ---------- Parseo SMILES ----------
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles.append(smiles)
                continue

            mol = self._standardize_molecule(mol)
            if mol is None:
                invalid_smiles.append(smiles)
                continue

            smiles_std = Chem.MolToSmiles(mol)

            # ---------- Descriptores ----------
            descriptors = calcular_descriptores(smiles_std)
            if descriptors is None:
                invalid_smiles.append(smiles)
                continue

            X_raw = np.asarray(descriptors, dtype=float).reshape(1, -1)
            mol_img = self._mol_to_base64(mol)

            # ---------- Predicción por blanco ----------
            for target, obj in self.models.items():
                clf = obj["model"]
                le = obj["label_encoder"]
                X_train = obj["X_train_preprocessed"]

                # Pipeline entrenado
                Xp = clf.named_steps["clean"].transform(X_raw)
                Xp = clf.named_steps["imputer"].transform(Xp)
                Xp = clf.named_steps["scaler"].transform(Xp)

                # Predicción
                class_index = list(le.classes_).index("Active")
                proba_active = clf.predict_proba(Xp)[0, class_index]

                pred_enc = clf.predict(Xp)[0]
                pred_label = le.inverse_transform([pred_enc])[0]

                # Dominio de aplicabilidad
                leverage, ad_flag = self._compute_leverage(Xp, X_train)

                all_results.append({
                    "SMILES": smiles,
                    "Target": target,
                    "Prediction": pred_label,
                    "Probability": proba_active,
                    "Leverage": leverage[0],
                    "AD_Flag": ad_flag[0],
                    "Molecule_Image": mol_img
                })

        results_df = pd.DataFrame(all_results)

        invalid_df = pd.DataFrame(
            {"Invalid_SMILES": invalid_smiles}
        )

        return results_df, invalid_df
