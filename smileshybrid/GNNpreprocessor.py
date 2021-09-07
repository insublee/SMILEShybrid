from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch

class GNNDatmodule:
    # def __init__(self, config):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.radius = 1
        self.dim = 50

    @staticmethod
    def create_atoms(mol, atom_dict):
        """Transform the atom types in a molecule (e.g., H, C, and O)
        into the indices (e.g., H=0, C=1, and O=2).
        Note that each atom index considers the aromaticity.
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)

    @staticmethod
    def create_ijbonddict(mol, bond_dict):
        """Create a dictionary, in which each key is a node ID
        and each value is the tuples of its neighboring node
        and chemical bond (e.g., single and double) IDs.
        """
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    @staticmethod
    def extract_fingerprints(radius, atoms, i_jbond_dict,
                            fingerprint_dict, edge_dict):
        """Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        """

        if (len(atoms) == 1) or (radius == 0):
            nodes = [fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_.append(fingerprint_dict[fingerprint])

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge_dict[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = nodes_
                i_jedge_dict = i_jedge_dict_

        return np.array(nodes)

    @staticmethod
    def split_dataset(dataset, ratio):
        """Shuffle and split a dataset."""
        np.random.seed(1234)  # fix the seed for shuffle.
        np.random.shuffle(dataset)
        n = int(ratio * len(dataset))
        return dataset[:n], dataset[n:]


    # def create_datasets(radius, device):
    # def create_datasets(self):
    def __call__(self, train_dataset, dev_dataset=None, test_dataset=None):
        """Initialize x_dict, in which each key is a symbol type
        (e.g., atom and chemical bond) and each value is its index.
        """
        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))

        def create_dataset(dataset):

            smileset = dataset['SMILES'].values.tolist()
            dataset_keys = dataset.keys()
            if 'labels' in dataset_keys:
                labelset = dataset['labels'].values.tolist()
            dataset = []
            if 'labels' in dataset_keys:
                for smiles, property in zip(smileset, labelset):
                    # smiles, property = data.strip().split() # smile, label
                    """Create each data with the above defined functions."""
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = self.create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = self.create_ijbonddict(mol, bond_dict)
                    fingerprints = self.extract_fingerprints(self.radius, atoms, i_jbond_dict,
                                                        fingerprint_dict, edge_dict)
                    adjacency = Chem.GetAdjacencyMatrix(mol)

                    """Transform the above each data of numpy
                    to pytorch tensor on a device (i.e., CPU or GPU).
                    """
                    fingerprints = torch.LongTensor(fingerprints).to(self.device)
                    adjacency = torch.FloatTensor(adjacency).to(self.device)

                    property = torch.FloatTensor([[float(property)]]).to(self.device)

                    dataset.append((fingerprints, adjacency, molecular_size, property))

                return dataset
            else: # test set (no label)
                for smiles in smileset:
                    # smiles, property = data.strip().split() # smile, label
                    """Create each data with the above defined functions."""
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = self.create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = self.create_ijbonddict(mol, bond_dict)
                    fingerprints = self.extract_fingerprints(self.radius, atoms, i_jbond_dict,
                                                        fingerprint_dict, edge_dict)
                    adjacency = Chem.GetAdjacencyMatrix(mol)

                    """Transform the above each data of numpy
                    to pytorch tensor on a device (i.e., CPU or GPU).
                    """
                    fingerprints = torch.LongTensor(fingerprints).to(self.device)
                    adjacency = torch.FloatTensor(adjacency).to(self.device)

                    dataset.append((fingerprints, adjacency, molecular_size))
                return dataset

        dataset_train = create_dataset(train_dataset)
        if dev_dataset is not None:
            dataset_dev = create_dataset(dev_dataset)
        if test_dataset is not None:
            dataset_test = create_dataset(test_dataset)

        N_fingerprints = len(fingerprint_dict)

# (dataset_train, dataset_dev, dataset_test, N_fingerprints) = GNNDatmodule()()

# a = [dataset_train, dataset_dev, N_fingerprints]
# print(a)