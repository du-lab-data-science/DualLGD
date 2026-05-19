from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import pathlib
import pickle
from contextlib import contextmanager
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

try:
    import fcntl
except ImportError:
    fcntl = None

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
from src.datasets.abstract_dataset import ATOM_TO_VALENCY, ATOM_TO_WEIGHT


def load_processed_graph_data(path):
    """
    Load preprocessed PyG objects with compatibility across PyTorch versions.

    PyTorch >= 2.6 changed torch.load default `weights_only=True`, which breaks
    loading serialized PyG Data/Batch objects. We explicitly request
    `weights_only=False` for local trusted processed dataset files.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose `weights_only`.
        return torch.load(path, map_location='cpu')


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def _is_cache_corruption_error(exc: Exception) -> bool:
    if isinstance(exc, (pickle.UnpicklingError, EOFError)):
        return True
    message = str(exc).lower()
    tokens = (
        'persistent id instruction',
        'pickle data was truncated',
        'failed finding central directory',
        'failed reading zip archive',
        'invalid load key',
    )
    return any(token in message for token in tokens)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return


@contextmanager
def _cache_init_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _get_int_env(name: str):
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_preprocess_n_jobs() -> int:
    env_n_jobs = _get_int_env('FP2MOL_PREPROCESS_NJOBS')
    if env_n_jobs is not None:
        return max(1, env_n_jobs)

    world_size = _get_int_env('WORLD_SIZE') or 1
    if world_size > 1:
        return 1

    slurm_cpus = _get_int_env('SLURM_CPUS_PER_TASK')
    if slurm_cpus is not None and slurm_cpus > 0:
        return max(1, min(slurm_cpus, 8))

    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def _get_preprocess_chunk_size() -> int:
    env_chunk = _get_int_env('FP2MOL_PREPROCESS_CHUNK_SIZE')
    if env_chunk is not None:
        return max(1, env_chunk)
    return 20000


def _append_processed_result(result, data_list, smiles_kept, filter_dataset: bool):
    if result is None:
        return
    if filter_dataset:
        data, smiles = result
        data_list.append(data)
        smiles_kept.append(smiles)
    else:
        data_list.append(result)

def _build_graph_from_mol(mol, types, bonds):
    """Build graph tensors (x, edge_index, edge_attr) from an RDKit mol.
    Returns (x, edge_index, edge_attr) or None if the molecule has unknown atoms or no bonds."""
    N = mol.GetNumAtoms()
    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in types:
            return None
        type_idx.append(types[symbol])
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]
    if len(row) == 0:
        return None
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
    return x, edge_index, edge_attr


def _filter_data(data, atom_decoder):
    """Filter a Data object by reconstructing the molecule and checking validity."""
    batch = getattr(data, 'batch', torch.zeros(data.x.size(0), dtype=torch.long))
    dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, batch)
    dense_data = dense_data.mask(node_mask, collapse=True)
    X, E = dense_data.X, dense_data.E
    if X.size(0) != 1:
        return None
    atom_types = X[0]
    edge_types = E[0]
    mol_reconstructed = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
    smiles = mol2smiles(mol_reconstructed)
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol_reconstructed, asMols=True, sanitizeFrags=True)
            if len(mol_frags) == 1:
                return (data, smiles)
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetMolFrags")
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
    return None


def process_single_inchi(args):
    """
    Process a single inchi string.

    Parameters:
        args: tuple of (i, inchi, types, bonds, morgan_r, morgan_nbits,
                           filter_dataset, pre_filter, pre_transform, atom_decoder)
    Returns:
        If filter_dataset is True: a tuple (data, smiles) if the molecule passes filtering,
            or None otherwise.
        Otherwise: the processed Data object (or None if it fails).
    """
    RDLogger.DisableLog('rdApp.*')

    #unpack args
    (i, inchi, types, bonds, morgan_r, morgan_nbits,
     filter_dataset, pre_filter, pre_transform, atom_decoder) = args

    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        # Remove stereochemistry information
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        graph = _build_graph_from_mol(mol, types, bonds)
        if graph is None:
            return None
        x, edge_index, edge_attr = graph
        fp = GetMorganFingerprintAsBitVect(mol, morgan_r, nBits=morgan_nbits)
        y = torch.tensor(np.asarray(fp, dtype=np.int8)).unsqueeze(0)
        inchi_canonical = Chem.MolToInchi(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, inchi=inchi_canonical)

        if filter_dataset:
            return _filter_data(data, atom_decoder)
        else:
            if pre_filter is not None and not pre_filter(data):
                return None
            if pre_transform is not None:
                data = pre_transform(data)
            return data
    except Exception as e:
        print(e)
        return None


atom_decoder = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
valency = [ATOM_TO_VALENCY.get(atom, 0) for atom in atom_decoder]

# Data sources: 
# HMDB: https://hmdb.ca/downloads
# DSSTox: https://clowder.edap-cluster.com/datasets/61147fefe4b0856fdc65639b#folderId=6616d85ce4b063812d70fc8f
# COCONUT: https://zenodo.org/records/13692394

class FP2MolDataset(InMemoryDataset):
    def __init__(self, stage, root, filter_dataset: bool, transform=None, pre_transform=None, pre_filter=None,
                 morgan_r=2, morgan_nBits=2048, dataset='hmdb'):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset

        self.morgan_r = morgan_r
        self.morgan_nbits = morgan_nBits
        self.dataset = dataset

        self._processed_dir = os.path.join(root, 'processed', f'morgan_r-{self.morgan_r}__morgan_nbits-{self.morgan_nbits}')
        self._raw_dir = os.path.join(root, 'preprocessed')

        if self.stage == 'train': self.file_idx = 0
        elif self.stage == 'val': self.file_idx = 1
        elif self.stage == 'test': self.file_idx = 1
        else: raise ValueError(f"Invalid stage {self.stage}")

        lock_path = os.path.join(self._processed_dir, '.cache_init.lock')
        with _cache_init_lock(lock_path):
            cache_paths = self._cache_paths()
            for attempt in range(2):
                try:
                    super().__init__(root, None, pre_transform, pre_filter)
                    self.data, self.slices = load_processed_graph_data(self.processed_paths[self.file_idx])
                    break
                except Exception as exc:
                    if attempt == 0 and _is_cache_corruption_error(exc):
                        for cache_path in cache_paths:
                            _safe_remove(cache_path)
                        continue
                    raise

    def _cache_paths(self):
        cache_files = list(to_list(self.processed_file_names))
        cache_files.extend(['pre_transform.pt', 'pre_filter.pt'])
        return [os.path.join(self._processed_dir, file_name) for file_name in cache_files]

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def raw_file_names(self):
        return [f"{self.dataset}_train.csv", f"{self.dataset}_val.csv"]

    @property
    def split_file_name(self):
        return [f"{self.dataset}_train.csv", f"{self.dataset}_val.csv"]


    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [os.path.join(self._raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        n_jobs = _get_preprocess_n_jobs()
        chunk_size = _get_preprocess_chunk_size()

        path = self.split_paths[self.file_idx]

        if not os.path.exists(self.processed_paths[self.file_idx]):
            data_list = []
            smiles_kept = []

            csv_iter = pd.read_csv(path, usecols=['inchi'], chunksize=chunk_size)

            offset = 0
            for chunk_df in csv_iter:
                if len(chunk_df) == 0:
                    continue

                inchi_list = chunk_df['inchi'].values

                if n_jobs == 1:
                    for i in tqdm(range(len(inchi_list)), desc="Processing inchi.....", total=len(inchi_list), leave=False):
                        result = process_single_inchi(
                            (offset + i, inchi_list[i], types, bonds, self.morgan_r, self.morgan_nbits,
                             self.filter_dataset, self.pre_filter, self.pre_transform, self.atom_decoder)
                        )
                        _append_processed_result(result, data_list, smiles_kept, self.filter_dataset)
                else:
                    args_list = [
                        (offset + i, inchi, types, bonds, self.morgan_r, self.morgan_nbits,
                         self.filter_dataset, self.pre_filter, self.pre_transform, self.atom_decoder)
                        for i, inchi in enumerate(inchi_list)
                    ]

                    with tqdm_joblib(tqdm(desc="Processing inchi.....", total=len(args_list), leave=False)):
                        results = Parallel(n_jobs=n_jobs)(delayed(process_single_inchi)(arg) for arg in args_list)

                    for result in tqdm(results, desc="Filtering graphs.....", total=len(results), leave=False):
                        _append_processed_result(result, data_list, smiles_kept, self.filter_dataset)

                offset += len(inchi_list)

            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

class FP2MolDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        self.dataset_name = cfg.dataset.dataset
        self._root_path = os.path.join(cfg.general.parent_dir, self.datadir, self.dataset_name)
        common_kwargs = dict(
            filter_dataset=self.filter_dataset,
            morgan_r=cfg.dataset.morgan_r,
            morgan_nBits=cfg.dataset.morgan_nbits,
            dataset=cfg.dataset.dataset,
        )
        datasets = {'train': FP2MolDataset(stage='train', root=self._root_path, **common_kwargs),
                    'val': FP2MolDataset(stage='val', root=self._root_path, **common_kwargs),
                    'test': FP2MolDataset(stage='test', root=self._root_path, **common_kwargs)}
        super().__init__(cfg, datasets)


class FP2Mol_infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = datamodule.dataset_name
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(self.atom_decoder)}
        self.valencies = valency
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = max(self.atom_weights.values())

        meta_files = dict(n_nodes=f'{datamodule._root_path}/stats/n_counts.txt',
                          node_types=f'{datamodule._root_path}/stats/atom_types.txt',
                          edge_types=f'{datamodule._root_path}/stats/edge_types.txt',
                          valency_distribution=f'{datamodule._root_path}/stats/valencies.txt')
        
        # n_nodes and valency_distribution are not transferrable between datatsets because of shape mismatches
        if cfg.dataset.stats_dir:
            meta_read = dict(n_nodes=f'{datamodule._root_path}/stats/n_counts.txt',
                          node_types=f'{cfg.dataset.stats_dir}/atom_types.txt',
                          edge_types=f'{cfg.dataset.stats_dir}/edge_types.txt',
                          valency_distribution=f'{datamodule._root_path}/stats/valencies.txt')
        else:
            meta_read = dict(n_nodes=f'{datamodule._root_path}/stats/n_counts.txt',
                          node_types=f'{datamodule._root_path}/stats/atom_types.txt',
                          edge_types=f'{datamodule._root_path}/stats/edge_types.txt',
                          valency_distribution=f'{datamodule._root_path}/stats/valencies.txt')
            

        self.n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None

        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())

        for k, v in meta_read.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = torch.tensor(np.loadtxt(v))
                setattr(self, k, meta[k])

        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None

        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
