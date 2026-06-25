import os
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# Amino acid
AMINO_ACIDS = ("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX: Dict[str, int] = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}

def encode_protein(seq: str, max_len: int = 1000) -> torch.LongTensor:
    encoded = [AA_TO_IDX.get(aa, 21) for aa in seq[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    return torch.tensor(padded, dtype=torch.long)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "implicit_valence": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ] if RDKIT_AVAILABLE else [],
}


def _one_hot(value, choices):
    enc = [0] * (len(choices) + 1)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        feat = (
            _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
            + _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
            + _one_hot(atom.GetImplicitValence(), ATOM_FEATURES["implicit_valence"])
            + _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            + _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
            + _one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
            + [int(atom.GetIsAromatic())]
        )
        node_feats.append(feat)

    x = torch.tensor(node_feats, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class DTADataset(Dataset):
    def __init__(
        self,
        drug_smiles: List[str],
        protein_seqs: List[str],
        labels: List[float],
        max_protein_len: int = 1000,
        cache_dir: Optional[str] = None,
    ):
        self.max_protein_len = max_protein_len
        self.labels = torch.tensor(labels, dtype=torch.float32)

        cache_path = Path(cache_dir) / "drug_graphs.pkl" if cache_dir else None
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                self.drug_graphs = pickle.load(f)
        else:
            print("Featurizing drugs (SMILES → graph)...")
            self.drug_graphs = [smiles_to_graph(s) for s in drug_smiles]
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.drug_graphs, f)

        print("Encoding protein sequences...")
        self.proteins = [
            encode_protein(seq, max_protein_len) for seq in protein_seqs
        ]

        valid_mask = [g is not None for g in self.drug_graphs]
        if not all(valid_mask):
            n_invalid = sum(1 for v in valid_mask if not v)
            print(f"Warning: loại bỏ {n_invalid} drug SMILES không hợp lệ")
            self.drug_graphs = [g for g, v in zip(self.drug_graphs, valid_mask) if v]
            self.proteins = [p for p, v in zip(self.proteins, valid_mask) if v]
            self.labels = self.labels[[i for i, v in enumerate(valid_mask) if v]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.drug_graphs[idx], self.proteins[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        drug_graphs, proteins, labels = zip(*batch)
        batched_drugs = Batch.from_data_list(list(drug_graphs))
        proteins = torch.stack(proteins)
        labels = torch.stack(labels)
        return batched_drugs, proteins, labels


# ─── Davis / KIBA Data Loading ────────────────────────────────────────────────

DAVIS_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/"
)
KIBA_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/"
)


def _download_if_needed(url: str, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def _load_affinity_matrix(path: Path) -> np.ndarray:
    import pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        return np.array(obj, dtype=np.float32)
    except Exception:
        pass
    try:
        return np.load(path, allow_pickle=True).astype(np.float32)
    except Exception:
        pass
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return np.array(obj, dtype=np.float32)


def load_davis(data_dir: str = "./data/raw") -> Tuple[List, List, List, pd.DataFrame]:
    base = Path(data_dir) / "davis"
    base.mkdir(parents=True, exist_ok=True)

    # Davis data files
    ligands_file = base / "ligands_can.txt"
    proteins_file = base / "proteins.txt"
    affinity_file = base / "Y"

    _download_if_needed(DAVIS_URL + "ligands_can.txt", ligands_file)
    _download_if_needed(DAVIS_URL + "proteins.txt", proteins_file)
    _download_if_needed(DAVIS_URL + "Y", affinity_file)

    # Load
    with open(ligands_file, encoding="utf-8") as f:
        ligands = eval(f.read())  # dict: name → SMILES
    with open(proteins_file, encoding="utf-8") as f:
        proteins = eval(f.read())  # dict: name → sequence

    Y = _load_affinity_matrix(affinity_file)

    drug_names = list(ligands.keys())
    protein_names = list(proteins.keys())

    drug_smiles_list = []
    protein_seqs_list = []
    labels_list = []
    meta_rows = []

    for di, dname in enumerate(drug_names):
        for pi, pname in enumerate(protein_names):
            affinity = Y[di][pi]
            if affinity != 0:  
                pkd = -np.log10(affinity / 1e9)
                drug_smiles_list.append(ligands[dname])
                protein_seqs_list.append(proteins[pname])
                labels_list.append(pkd)
                meta_rows.append({
                    "drug_id": di,
                    "drug_name": dname,
                    "protein_id": pi,
                    "protein_name": pname,
                    "kinase_family": _infer_kinase_family(pname),
                    "label": pkd,
                })

    meta_df = pd.DataFrame(meta_rows)
    print(f"Davis: {len(labels_list)} samples, "
          f"{len(drug_names)} drugs, {len(protein_names)} proteins")
    return drug_smiles_list, protein_seqs_list, labels_list, meta_df


def load_kiba(data_dir: str = "./data/raw") -> Tuple[List, List, List, pd.DataFrame]:
    base = Path(data_dir) / "kiba"
    base.mkdir(parents=True, exist_ok=True)

    ligands_file = base / "ligands_can.txt"
    proteins_file = base / "proteins.txt"
    affinity_file = base / "Y"

    _download_if_needed(KIBA_URL + "ligands_can.txt", ligands_file)
    _download_if_needed(KIBA_URL + "proteins.txt", proteins_file)
    _download_if_needed(KIBA_URL + "Y", affinity_file)

    with open(ligands_file, encoding="utf-8") as f:
        ligands = eval(f.read())
    with open(proteins_file, encoding="utf-8") as f:
        proteins = eval(f.read())

    Y = _load_affinity_matrix(affinity_file)

    drug_names = list(ligands.keys())
    protein_names = list(proteins.keys())

    drug_smiles_list, protein_seqs_list, labels_list, meta_rows = [], [], [], []

    for di, dname in enumerate(drug_names):
        for pi, pname in enumerate(protein_names):
            affinity = Y[di][pi]
            if not np.isnan(affinity):
                drug_smiles_list.append(ligands[dname])
                protein_seqs_list.append(proteins[pname])
                labels_list.append(float(affinity))
                meta_rows.append({
                    "drug_id": di,
                    "drug_name": dname,
                    "protein_id": pi,
                    "protein_name": pname,
                    "kinase_family": _infer_kinase_family(pname),
                    "label": float(affinity),
                })

    meta_df = pd.DataFrame(meta_rows)
    print(f"KIBA: {len(labels_list)} samples, "
          f"{len(drug_names)} drugs, {len(protein_names)} proteins")
    return drug_smiles_list, protein_seqs_list, labels_list, meta_df


def load_dataset(dataset: str, data_dir: str = "./data/raw"):
    if dataset.lower() == "davis":
        return load_davis(data_dir)
    elif dataset.lower() == "kiba":
        return load_kiba(data_dir)
    else:
        raise ValueError(f"Error")


# Kinase family 

KINASE_FAMILY_KEYWORDS = {
    "CDK": "CDK",
    "MAPK": "MAPK",
    "ERK": "MAPK",
    "MEK": "MAPK",
    "JAK": "JAK",
    "SRC": "SRC",
    "ABL": "ABL",
    "VEGFR": "VEGFR",
    "EGFR": "EGFR",
    "PDGFR": "PDGFR",
    "FGFR": "FGFR",
    "PI3K": "PI3K",
    "AKT": "PI3K",
    "GSK": "GSK",
    "PLK": "PLK",
    "Aurora": "Aurora",
    "CHK": "CHK",
    "WEE": "WEE",
}


def _infer_kinase_family(protein_name: str) -> str:
    """Heuristic: suy ra kinase family từ tên protein."""
    for keyword, family in KINASE_FAMILY_KEYWORDS.items():
        if keyword.upper() in protein_name.upper():
            return family
    return "OTHER"
