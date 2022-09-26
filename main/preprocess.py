from collections import defaultdict

import numpy as np

from rdkit import Chem

import torch
## partial of the preprocessing codes are obtained from QDF

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


def create_ijbonddict(mol, bond_dict, bond_fingerprint_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    num_bonds = len(mol.GetBonds())
    bond_adjacency = np.zeros((num_bonds, num_bonds))
    i_jbond_dict = defaultdict(lambda: [])
    bond_adj = defaultdict(lambda: [])
    bond_index = Chem.GetAdjacencyMatrix(mol)
    nodes = []
    for i_bond, b in enumerate(mol.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
        tmp = bond_fingerprint_dict[(mol.GetAtoms()[i].GetSymbol(),mol.GetAtoms()[i].GetSymbol(), bond)]
        nodes.append(tmp)

        bond_adj[i].append(i_bond)
        bond_adj[j].append(i_bond)
        bond_index[i,j] = tmp
        bond_index[j, i] = tmp

    for atom in bond_adj.keys():
        related_bonds = bond_adj[atom]
        for i in related_bonds:
            for j in related_bonds:
                if i == j:
                    continue
                bond_adjacency[i,j] = 1
                # this value can be set as angle (float)

    return i_jbond_dict, bond_adjacency, np.array(nodes), bond_index


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Notice that the fingerprints ID for each atoms depends on itself, as well as its nearby atoms.
    """
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs. Here the radius is set
             as 0 so that this step is skipped
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

def split_dataset(dataset, ratio):
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(task, dataset, radius, device):

    dir_dataset = '../dataset/' + task + '/' + dataset + '/'

    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    bond_fingerprint_dict = defaultdict(lambda: len(bond_fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):

        print(filename)

        """Load a dataset."""
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles."""
        data_original = [data for data in data_original
                         if '.' not in data.split()[0]]

        dataset = []

        for data in data_original:
            try:
                smiles, property = data.strip().split()
            except:
                print(data)
                continue

            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict, bond_adjacency, bond_fingerprints, bond_index = create_ijbonddict(mol, bond_dict, bond_fingerprint_dict)# for atom's bond-angle graph
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)# for atom's atom-bond graph

            fingerprints = torch.LongTensor(fingerprints).to(device)
            bond_fingerprints = torch.LongTensor(bond_fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            bond_index = torch.LongTensor(bond_index).to(device)
            if task == 'classification':
                property = torch.LongTensor([int(property)]).to(device)
            if task == 'regression':
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, bond_fingerprints, bond_adjacency, bond_index, molecular_size, property))

        return dataset

    dataset_train = create_dataset('data_train.txt')
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset('data_test.txt')
    np.save('atom_dict.npy', dict(atom_dict), )
    np.save('bond_dict.npy', dict(bond_dict), )
    np.save('fingerprint_dict.npy', dict(fingerprint_dict), )
    np.save('bond_fingerprint_dict.npy', dict(bond_fingerprint_dict), )
    np.save('edge_dict.npy', dict(edge_dict), )
    N_fingerprints = len(fingerprint_dict)
    N_bond_fingerprints = len(bond_fingerprint_dict)

    return dataset_train, dataset_dev, dataset_test, N_fingerprints, N_bond_fingerprints
