import os
from functools import lru_cache
from typing import List

import numpy as np
import torch_geometric.utils
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

def cfg_to_dict(cfg):
    return omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _resolve_dataset_paths(cfg: DictConfig) -> None:
    """Resolve dataset path fields against Hydra original cwd."""
    path_fields = (
        "datadir",
        "split_file",
        "subform_folder",
        "labels_file",
        "spec_folder",
        "stats_dir",
    )

    with open_dict(cfg):
        for field in path_fields:
            raw = cfg.dataset.get(field)
            if raw is None or not isinstance(raw, str):
                continue
            cfg.dataset[field] = hydra.utils.to_absolute_path(raw)

def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    def _add_missing_keys(target_cfg, source_cfg):
        with open_dict(target_cfg):
            for key, val in source_cfg.items():
                if key not in target_cfg.keys():
                    target_cfg[key] = OmegaConf.create(OmegaConf.to_container(val, resolve=False)) if OmegaConf.is_config(val) else val
                    continue

                target_val = target_cfg[key]
                if OmegaConf.is_dict(val) and OmegaConf.is_dict(target_val):
                    _add_missing_keys(target_val, val)

    _add_missing_keys(cfg, saved_cfg)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def is_valid(mol):
    smiles = mol2smiles(mol)
    if smiles is None:
        return False

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        return False
    if len(mol_frags) > 1:
        return False
    
    return True

def inchi_to_fingerprint(inchi: str, nbits: int = 2048, radius=3) -> np.ndarray:
    """get_morgan_fp."""

    mol = mol_from_inchi_cached(inchi)
    if mol is None:
        return np.zeros((0,), dtype=np.uint8)

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint

def tanimoto_sim(x: np.ndarray, y: np.ndarray) -> List[float]:
    # Calculate tanimoto distance with binary fingerprint
    intersect_mat = x & y
    union_mat = x | y

    intersection = intersect_mat.sum(-1)
    union = union_mat.sum(-1)

    ### I took the reciprocal here so instead of tanimoto sim, it became
    # distance. Could have just made negative but
    # sklearn doesn't accept negative distance matrices
    output = intersection / union
    return output

def cosine_sim(x: np.ndarray, y: np.ndarray) -> List[float]:
    # Calculate cosine similarity with binary fingerprint
    dot_product = np.dot(x, y)

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    output = dot_product / (norm_x * norm_y)
    return output

try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'


@lru_cache(maxsize=16384)
def _cached_mol_from_inchi(inchi: str):
    if not inchi:
        return None
    return Chem.MolFromInchi(inchi)


def mol_from_inchi_cached(inchi: str):
    """Return a cloned RDKit molecule from an InChI cache.

    The clone avoids accidental in-place mutation of the cached object.
    """
    mol = _cached_mol_from_inchi(inchi)
    if mol is None:
        return None
    return Chem.Mol(mol)

def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = mol_from_inchi_cached(inchi)
    if mol is None:
        return None
    
    try:
        if _RD_TAUTOMER_CANONICALIZER == 'v1':
            _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
            mol = _molvs_t.canonicalize(mol)
        else:
            _te = TautomerEnumerator()
            mol = _te.Canonicalize(mol)
    except Chem.rdchem.KekulizeException:
        print(f"Can't kekulize molecule during canonicalization for InChI: {inchi}")
        return None
    except Exception as e:
        print(f"Error during canonicalization for InChI: {inchi}: {e}")
        return None
    
    return mol
