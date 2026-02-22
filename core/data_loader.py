"""
HemaVision Data Loader
━━━━━━━━━━━━━━━━━━━━━━
Patient-level data preprocessing for AML detection.

Handles the critical ONE-TO-MANY relationship:
  1 patient → N cell images → N training rows

Splitting Strategy:
  - Split by PATIENT (not by image) to prevent data leakage
  - Train 70% / Val 10% / Test 20% of patients
  - All images from one patient stay in the same split

Author: Firoj 
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.config import HemaVisionConfig, get_config

logger = logging.getLogger(__name__)


class AMLDataPreprocessor:
    """
    Preprocesses the Munich AML-Cytomorphology dataset.

    This class handles:
    1. Loading patient metadata from CSV
    2. Discovering cell images and mapping them to patients
    3. Creating a unified DataFrame (image_path, features, label)
    4. Patient-level train/val/test splitting
    5. Feature normalization (age, sex, genetic markers)

    Architecture:
    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
    │ patient_data │ ──▶ │   Unified     │ ──▶ │ Train/Val/   │
    │    .csv      │     │  DataFrame    │     │ Test Splits  │
    └──────────────┘     └───────────────┘     └──────────────┘
           ▲                     │
    ┌──────┴───────┐            │
    │  images/     │            ▼
    │  ├─ P001_*   │     ┌──────────────┐
    │  ├─ P002_*   │     │  Normalized  │
    │  └─ ...      │     │  Features    │
    └──────────────┘     └──────────────┘
    """

    # Class labels mapping
    #
    # IMPORTANT: The Kaggle uploads of this dataset (binilj04, umarsani1605)
    # renamed the BLA (Blast) folder to MYO. Evidence:
    #   - Original Matek et al. 2019 paper: BLA = 3,268 images
    #   - Kaggle upload: MYO = 3,268 images (identical count)
    #   - All other 14 classes have identical counts to the paper
    #   - "Myelocyte" (real MYO) had 0 images in the original 15-class dataset
    #   - Total matches exactly: 18,365 images
    # Therefore MYO → 1 (AML positive) in all Kaggle versions.
    #
    AML_CLASSES = {
        # AML blast cells (malignant)
        "BLA": 1,  # Blast (original TCIA class name)
        "MYO": 1,  # Blast (renamed to MYO in Kaggle uploads)
        # Normal / non-malignant cell types
        "ART": 0,  # Artifact
        "BAS": 0,  # Basophil
        "EBO": 0,  # Erythroblast
        "EOS": 0,  # Eosinophil
        "KSC": 0,  # Smudge cell
        "LYA": 0,  # Lymphocyte (atypical)
        "LYT": 0,  # Lymphocyte (typical)
        "MMZ": 0,  # Metamyelocyte
        "MOB": 0,  # Monocyte (blast-like)
        "MON": 0,  # Monocyte
        "MYB": 0,  # Myelocyte (basophilic)
        "NGB": 0,  # Band neutrophil
        "NGS": 0,  # Segmented neutrophil
        "NIF": 0,  # Not identifiable
        "OTH": 0,  # Other
        "PEB": 0,  # Proerythroblast
        "PLM": 0,  # Plasma cell
        "PMB": 0,  # Promyelocyte (bilobed)
        "PMO": 0,  # Promyelocyte
    }

    def __init__(self, config: Optional[HemaVisionConfig] = None):
        """
        Initialize the data preprocessor.

        Args:
            config: HemaVision configuration object. Uses default if None.
        """
        self.config = config or get_config()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Will be populated during processing
        self.unified_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.tabular_feature_names: List[str] = []
        self.num_tabular_features: int = 0

    def create_unified_dataframe(self) -> pd.DataFrame:
        """
        Create the master DataFrame linking images to patient metadata.

        Handles ONE-TO-MANY relationship:
        - Patient P001 has 50 cell images → 50 rows in final DataFrame
        - Each row has: [image_path, age, sex, genetic_subtype, label]

        The method supports two modes:
        1. With patient_data.csv → merges clinical data with image paths
        2. Without CSV → infers labels from directory structure / filenames

        Returns:
            pd.DataFrame: Unified dataset ready for training
        """
        images_dir = self.config.paths.images_dir
        patient_csv = self.config.paths.patient_csv

        logger.info(f"Scanning images from: {images_dir}")
        if images_dir != self.config.paths.data_root:
            logger.info(f"  (auto-detected from data_root: {self.config.paths.data_root})")

        # Discover all images
        image_records = self._discover_images(images_dir)

        if len(image_records) == 0:
            raise FileNotFoundError(
                f"No images found in {images_dir}. "
                f"Please check the dataset path in config."
            )

        images_df = pd.DataFrame(image_records)
        logger.info(f"Discovered {len(images_df)} images from "
                     f"{images_df['patient_id'].nunique()} patients")

        # Try to load patient metadata CSV
        if patient_csv.exists():
            logger.info(f"Loading patient metadata from: {patient_csv}")
            patient_df = pd.read_csv(patient_csv)
            patient_df = self._normalize_patient_columns(patient_df)

            # Merge images with patient data
            self.unified_df = images_df.merge(
                patient_df,
                on="patient_id",
                how="left"
            )

            # Fill missing patient data with defaults
            self._fill_missing_metadata()
        else:
            logger.warning(
                f"Patient CSV not found at {patient_csv}. "
                f"Using synthetic clinical data for demonstration."
            )
            self.unified_df = self._generate_synthetic_metadata(images_df)

        # Ensure binary label exists
        if "label" not in self.unified_df.columns:
            self.unified_df["label"] = self.unified_df["cell_type"].map(
                self.AML_CLASSES
            ).fillna(0).astype(int)

        logger.info(
            f"Unified DataFrame: {len(self.unified_df)} rows, "
            f"Label distribution: "
            f"{self.unified_df['label'].value_counts().to_dict()}"
        )

        return self.unified_df

    def _discover_images(self, images_dir: Path) -> List[Dict]:
        """
        Discover all cell images and extract metadata from filenames/dirs.

        Supports two directory structures:
        1. Flat: images/PatientXXX_CellType_NNN.jpg
        2. Nested: images/CellType/PatientXXX_NNN.jpg

        Returns:
            List of dicts with image_path, patient_id, cell_type
        """
        records = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

        if not images_dir.exists():
            logger.warning(f"Images directory does not exist: {images_dir}")
            return records

        # Check if nested structure (subdirectories by cell type)
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Nested structure: images/CellType/filename.ext
            for subdir in sorted(subdirs):
                cell_type = subdir.name.upper()
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in valid_extensions:
                        patient_id = self._extract_patient_id(img_path.stem)
                        records.append({
                            "image_path": str(img_path),
                            "patient_id": patient_id,
                            "cell_type": cell_type,
                        })
        else:
            # Flat structure: images/PatientXXX_CellType_NNN.ext
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    patient_id = self._extract_patient_id(img_path.stem)
                    cell_type = self._extract_cell_type(img_path.stem)
                    records.append({
                        "image_path": str(img_path),
                        "patient_id": patient_id,
                        "cell_type": cell_type,
                    })

        return records

    @staticmethod
    def _extract_patient_id(filename: str) -> str:
        """
        Extract patient ID from filename.

        Examples:
            "Patient001_Cell_001" → "Patient001"
            "BLA_Patient005_123"  → "Patient005"
            "unknown_file"        → "Unknown_Patient"
        """
        # Try pattern: PatientXXX or PXXX
        match = re.search(r"(Patient\d+|P\d+)", filename, re.IGNORECASE)
        if match:
            return match.group(1)

        # Try extracting first segment before underscore as ID
        parts = filename.split("_")
        if len(parts) >= 2:
            return parts[0]

        return "Unknown_Patient"

    @staticmethod
    def _extract_cell_type(filename: str) -> str:
        """Extract cell type abbreviation from filename."""
        known_types = [
            "BLA", "ART", "BAS", "EBO", "EOS", "KSC", "LYA", "LYT",
            "MMZ", "MOB", "MON", "MYB", "MYO", "NGB", "NGS", "NIF",
            "OTH", "PEB", "PLM", "PMB", "PMO"
        ]
        upper = filename.upper()
        for ct in known_types:
            if ct in upper:
                return ct
        return "OTH"

    def _normalize_patient_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from various CSV formats."""
        col_mapping = {
            "Patient_ID": "patient_id",
            "patient_id": "patient_id",
            "PatientID": "patient_id",
            "Age": "age",
            "age": "age",
            "Sex": "sex",
            "sex": "sex",
            "Gender": "sex",
            "Diagnosis": "diagnosis",
            "diagnosis": "diagnosis",
            "Genetic_Subtype": "genetic_subtype",
            "genetic_subtype": "genetic_subtype",
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
        return df

    def _fill_missing_metadata(self):
        """Fill any missing metadata with sensible defaults."""
        df = self.unified_df

        if "age" in df.columns:
            df["age"] = df["age"].fillna(df["age"].median())
        else:
            df["age"] = 60  # Default median age

        if "sex" in df.columns:
            df["sex"] = df["sex"].fillna("Unknown")
        else:
            df["sex"] = "Unknown"

        if "genetic_subtype" in df.columns:
            df["genetic_subtype"] = df["genetic_subtype"].fillna("None")
        else:
            df["genetic_subtype"] = "None"

        self.unified_df = df

    def _generate_synthetic_metadata(self, images_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic clinical metadata for demonstration purposes.
        Used when no patient_data.csv is available.

        IMPORTANT: When "patients" are really cell-type folders (few unique
        IDs), we randomize per IMAGE to prevent the tabular MLP from
        memorizing a unique feature pattern per cell type (= per label).
        """
        rng = np.random.RandomState(self.config.training.random_seed)
        unique_patients = images_df["patient_id"].unique()
        num_patients = len(unique_patients)
        num_images = len(images_df)

        # If pseudo-patients (cell-type folders), randomize per image
        # to prevent tabular features from leaking the label.
        per_image = num_patients < 20
        n = num_images if per_image else num_patients

        if per_image:
            logger.warning(
                f"Only {num_patients} pseudo-patients detected (cell-type folders). "
                f"Generating synthetic metadata per IMAGE to prevent label leakage."
            )

        meta = pd.DataFrame({
            "age": rng.normal(55, 15, n).clip(18, 95).astype(int),
            "sex": rng.choice(["Male", "Female"], n),
            "npm1_mutated": rng.choice([0, 1], n, p=[0.65, 0.35]),
            "flt3_mutated": rng.choice([0, 1], n, p=[0.7, 0.3]),
            "genetic_other": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "genetic_subtype": rng.choice(
                ["NPM1", "FLT3-ITD", "CEBPA", "t(8;21)", "inv(16)", "None"],
                n,
                p=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40]
            ),
        })

        if per_image:
            # Attach directly to images (same row order)
            df = images_df.reset_index(drop=True)
            for col in meta.columns:
                df[col] = meta[col].values
        else:
            # Per-patient: merge as before
            meta["patient_id"] = unique_patients
            df = images_df.merge(meta, on="patient_id", how="left")

        # Generate labels from cell types
        df["label"] = df["cell_type"].map(self.AML_CLASSES).fillna(0).astype(int)

        return df

    def prepare_features(self) -> Tuple[List[str], int]:
        """
        Encode and normalize clinical/tabular features.

        Transforms:
        - age: StandardScaler normalization
        - sex: One-hot encoding (Male=1, Female=0)
        - genetic_subtype: One-hot encoding
        - npm1_mutated, flt3_mutated, genetic_other: Binary (already 0/1)

        Returns:
            Tuple of (feature_column_names, num_features)
        """
        if self.unified_df is None:
            raise RuntimeError("Call create_unified_dataframe() first.")

        df = self.unified_df.copy()

        # Encode sex → binary
        df["sex_encoded"] = (df["sex"].str.lower() == "male").astype(float)

        # Normalize age
        df["age_normalized"] = self.scaler.fit_transform(
            df[["age"]].astype(float)
        )

        # One-hot encode genetic subtype
        if "genetic_subtype" in df.columns:
            genetic_dummies = pd.get_dummies(
                df["genetic_subtype"], prefix="gene"
            ).astype(float)
            df = pd.concat([df, genetic_dummies], axis=1)
            genetic_cols = list(genetic_dummies.columns)
        else:
            genetic_cols = []

        # Ensure binary mutation columns exist
        for col in ["npm1_mutated", "flt3_mutated", "genetic_other"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)

        # Define feature columns
        self.tabular_feature_names = (
            ["age_normalized", "sex_encoded",
             "npm1_mutated", "flt3_mutated", "genetic_other"]
            + genetic_cols
        )
        self.num_tabular_features = len(self.tabular_feature_names)

        # Verify all feature columns exist and are numeric
        for col in self.tabular_feature_names:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        self.unified_df = df
        logger.info(
            f"Prepared {self.num_tabular_features} tabular features: "
            f"{self.tabular_feature_names}"
        )
        return self.tabular_feature_names, self.num_tabular_features

    def split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data to prevent data leakage, with adaptive strategy.

        Strategy Selection:
        ┌──────────────────────────────────────────────────────────┐
        │  If real patient IDs available (≥20 patients):           │
        │    → Patient-level split (prevents data leakage)         │
        │  If pseudo-patients (e.g. cell-type folders only):       │
        │    → Stratified image-level split (ensures class          │
        │      balance in every split)                             │
        └──────────────────────────────────────────────────────────┘

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.unified_df is None:
            raise RuntimeError("Call create_unified_dataframe() first.")

        seed = self.config.training.random_seed
        train_r = self.config.training.train_ratio
        val_r = self.config.training.val_ratio
        test_r = self.config.training.test_ratio

        patients = self.unified_df["patient_id"].unique()
        num_patients = len(patients)

        # Decide splitting strategy:
        # If patients are really just cell-type folders (few "patients")
        # or a split would leave a class with 0 samples, use stratified
        # image-level splitting instead.
        use_patient_split = self._can_use_patient_split(patients, test_r, val_r, seed)

        if use_patient_split:
            self._split_by_patient(patients, train_r, val_r, test_r, seed)
        else:
            self._split_by_image_stratified(train_r, val_r, test_r, seed)

        # Compute class weights from training set for weighted loss
        train_labels = self.train_df["label"]
        pos_count = train_labels.sum()
        neg_count = len(train_labels) - pos_count
        self.pos_weight = neg_count / max(pos_count, 1)

        logger.info(
            f"Split complete:\n"
            f"  Train: {self.train_df['patient_id'].nunique()} patients, "
            f"{len(self.train_df)} images "
            f"(pos_rate={self.train_df['label'].mean():.3f})\n"
            f"  Val:   {self.val_df['patient_id'].nunique()} patients, "
            f"{len(self.val_df)} images "
            f"(pos_rate={self.val_df['label'].mean():.3f})\n"
            f"  Test:  {self.test_df['patient_id'].nunique()} patients, "
            f"{len(self.test_df)} images "
            f"(pos_rate={self.test_df['label'].mean():.3f})\n"
            f"  Positive weight (class imbalance): {self.pos_weight:.2f}"
        )

        return self.train_df, self.val_df, self.test_df

    def _can_use_patient_split(
        self, patients: np.ndarray, test_r: float, val_r: float, seed: int
    ) -> bool:
        """Check whether patient-level splitting keeps both classes in every split."""
        if len(patients) < 20:
            logger.warning(
                f"Only {len(patients)} unique patient IDs found — too few for "
                f"reliable patient-level splitting. Using stratified image-level split."
            )
            return False

        # Simulate the split and verify class coverage
        try:
            tv, test_p = train_test_split(patients, test_size=test_r, random_state=seed)
            val_rel = val_r / (1 - test_r)
            train_p, val_p = train_test_split(tv, test_size=val_rel, random_state=seed)

            for name, pset in [("val", val_p), ("test", test_p)]:
                subset = self.unified_df[self.unified_df["patient_id"].isin(pset)]
                if subset["label"].nunique() < 2:
                    logger.warning(
                        f"Patient-level split would give {name} set only "
                        f"class(es) {subset['label'].unique().tolist()}. "
                        f"Falling back to stratified image-level split."
                    )
                    return False
        except ValueError:
            return False

        return True

    def _split_by_patient(
        self, patients, train_r, val_r, test_r, seed
    ):
        """Patient-level splitting (no data leakage)."""
        logger.info(f"Splitting {len(patients)} patients into "
                     f"train({train_r})/val({val_r})/test({test_r}) "
                     f"[patient-level]")

        train_val_patients, test_patients = train_test_split(
            patients, test_size=test_r, random_state=seed,
        )
        val_relative_size = val_r / (train_r + val_r)
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_relative_size, random_state=seed,
        )

        self.train_df = self.unified_df[
            self.unified_df["patient_id"].isin(train_patients)
        ].reset_index(drop=True)
        self.val_df = self.unified_df[
            self.unified_df["patient_id"].isin(val_patients)
        ].reset_index(drop=True)
        self.test_df = self.unified_df[
            self.unified_df["patient_id"].isin(test_patients)
        ].reset_index(drop=True)

    def _split_by_image_stratified(self, train_r, val_r, test_r, seed):
        """Stratified image-level splitting ensuring both classes in every split."""
        logger.info(
            f"Using STRATIFIED IMAGE-LEVEL split "
            f"train({train_r})/val({val_r})/test({test_r}) "
            f"to ensure class balance in all splits."
        )

        labels = self.unified_df["label"]

        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            self.unified_df.index,
            test_size=test_r,
            random_state=seed,
            stratify=labels,
        )

        # Second split: separate validation from training
        val_relative_size = val_r / (train_r + val_r)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative_size,
            random_state=seed,
            stratify=labels.loc[train_val_idx],
        )

        self.train_df = self.unified_df.loc[train_idx].reset_index(drop=True)
        self.val_df = self.unified_df.loc[val_idx].reset_index(drop=True)
        self.test_df = self.unified_df.loc[test_idx].reset_index(drop=True)

        return self.train_df, self.val_df, self.test_df

    def get_class_weights(self) -> float:
        """Return the positive class weight for weighted BCE loss."""
        if not hasattr(self, "pos_weight"):
            raise RuntimeError("Call split_data() first.")
        return self.pos_weight

    def get_split_summary(self) -> Dict:
        """Return a summary of the data splits for logging."""
        if self.train_df is None:
            raise RuntimeError("Call split_data() first.")

        def _summarize(df, name):
            return {
                "name": name,
                "num_patients": df["patient_id"].nunique(),
                "num_images": len(df),
                "label_distribution": df["label"].value_counts().to_dict(),
                "positive_rate": float(df["label"].mean()),
            }

        return {
            "train": _summarize(self.train_df, "train"),
            "val": _summarize(self.val_df, "val"),
            "test": _summarize(self.test_df, "test"),
            "num_tabular_features": self.num_tabular_features,
            "tabular_features": self.tabular_feature_names,
        }
