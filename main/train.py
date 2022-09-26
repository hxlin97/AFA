import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import preprocess as pp

# the AFA model
class AFA(nn.Module):
    def __init__(self, N_fingerprints, N_bond_fingerprints, dim, layer_hidden, layer_output):
        super(AFA, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.embed_bond_fingerprint = nn.Embedding(N_bond_fingerprints, dim)
        # you can revise this layer to contain more information
        # this dim corresponds to the TN states of each atom
        self.W_fingerprint = nn.Linear(dim, dim)
        self.W_bond_fingerprint = nn.Linear(dim, dim)
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value, device="cuda:0"):
        """Only for batch process"""
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def pad_MPO(self, matrices, pad_value, device="cuda:0"):
        """Only for batch process"""
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.LongTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def to_output(self, x):
        for l in range(len(self.W_output)):
            x = self.W_output[l](x)
            x = torch.relu(x)
        outputs = self.W_property(x)
        return outputs

    def update_MPS(self, adjacencies, MPSs):
        contri_MPS = torch.relu(self.W_fingerprint(MPSs))
        # the contribution from nearby atoms
        # you can repeat this step for n times, so that n-nearest atoms contributes to the atom
        return MPSs + torch.matmul(adjacencies, contri_MPS)

    def update_MPO(self, bond_adjacencies, MPOs):
        contri_MPO = torch.relu(self.W_bond_fingerprint(MPOs))
        # the contribution from nearby atoms
        # you can repeat this step for n times, so that n-nearest atoms contributes to the atom
        return MPOs + torch.matmul(bond_adjacencies, contri_MPO)

    def tn_contraction(self, inputs):
        """Construct the MPS for each atom and then contract them"""
        fingerprints, adjacencies, bond_fingerprints, bond_adjacencies, bond_index, molecular_sizes = inputs

        fingerprints = torch.cat(fingerprints)
        bond_fingerprints = torch.cat(bond_fingerprints)
        adjacencies = self.pad(adjacencies, 0) # 0 means false
        bond_index = self.pad_MPO(bond_index, 0)
        MPSs = self.embed_fingerprint(fingerprints)
        MPOs = self.embed_bond_fingerprint(bond_index)
        MPSs = self.update_MPS(adjacencies, MPSs)
        MPSs = F.normalize(MPSs, 2, 1)
        MPOs = F.normalize(MPOs, 2, 1)

        tmp = torch.einsum("ac,abc->cb", MPSs, MPOs)
        # tmp = torch.einsum("ab,ac->bc", MPSs, adjacencies)
        info_sets = torch.split(tmp, molecular_sizes,dim=1)
        fv_sets = torch.split(MPSs, molecular_sizes)
        TN_results = [torch.einsum("bc,cb->b", i, j) for (i,j) in zip(info_sets, fv_sets)]
        TN_results = torch.stack(TN_results)
        return TN_results

    def forward(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])
        if train:
            molecular_vectors = self.tn_contraction(inputs)
            predicted_values = self.to_output(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss

        else:
            with torch.no_grad():
                molecular_vectors = self.tn_contraction(inputs)
                predicted_values = self.to_output(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values



class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values = self.model.forward(
                                               data_batch, train=False)
            SAE += sum(np.abs(predicted_values-correct_values))
        MAE = SAE / N  # mean absolute error.
        return MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

if __name__ == "__main__":

    (task, dataset, radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = sys.argv[1:]
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [radius, dim, layer_hidden, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('Error: GPU required')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    (dataset_train, dataset_dev, dataset_test,
     N_fingerprints, N_bond_fingerprints) = pp.create_datasets(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = AFA(
            N_fingerprints, N_bond_fingerprints, dim, layer_hidden, layer_output).to(device)
    print(" the number of fingerprints are {}".format(N_fingerprints))
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/result--' + setting + '.txt'
    if task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test'
    if task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        prediction_dev = tester.test(dataset_dev)
        prediction_test = tester.test(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     prediction_dev, prediction_test]))
        tester.save_result(result, file_result)

        print(result)
    torch.save(model, f"{dataset}.pt")