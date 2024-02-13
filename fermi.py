import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FERMI(torch.nn.Module):
    def __init__(self, X_train, X_test, Y_train, Y_test, S_train, S_test, device, lam=0.1):
        super(FERMI, self).__init__()

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.S_train = S_train

        self.n = X_train.shape[0] # num of training data
        self.d = X_train.shape[1] # dimension of data
        self.m = Y_train.shape[1] # num of labels
        self.k = S_train.shape[1] # dim of encoded sensitive attributes, e.g. k = 2 if one considers a binary attribute.

        self.W = nn.Parameter(torch.zeros((self.k, self.m)))

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 1)

        p_s = self.S_train.sum(axis=0) / self.n

        final_entries = []
        for item in p_s:
            final_entries.append(1.0 / np.sqrt(item))

        self.P_s = np.diag(p_s)
        self.P_s_sqrt_inv = torch.from_numpy(np.diag(final_entries)).double()

        self.lam = lam
        self.device = device

        self.summation = 0
        self.term = None
        self.term1 = None
        self.term2 = None
        self.term3 = None
        
        self.summation2 = 0

    def forward(self, X):
        return self.fc2(F.relu(self.fc1(X)))
    
    def fairness_regularizer(self, X, S): # implementation based on the paper
        current_batch_size = X.shape[0]
        self.summation = 0

        Y_hat = torch.sigmoid(self.forward(X))

        for i in range(current_batch_size):
            self.term = Y_hat[i] * Y_hat[i] * torch.matmul(self.W.double(), torch.t(self.W.double()))
            self.summation += -torch.trace(self.term)
        
            self.term1 = Y_hat[i] * self.W.double()
            self.term2 = torch.matmul(self.term1.double(), torch.t(S[i]).unsqueeze(0).double())
            self.term3 = torch.matmul(self.term2.double(), self.P_s_sqrt_inv.to(self.device))
            self.summation += 2 * torch.trace(self.term3) - 1            
        
        return self.lam * (self.summation / current_batch_size)

    def fairness_regularizer2(self, X, S): # implementation for faster execution time 
        current_batch_size = X.shape[0]
        self.summation2 = 0

        Y_hat = torch.sigmoid(self.forward(X))

        a1 = torch.matmul(Y_hat.T, Y_hat)
        a2 = a1 * torch.matmul(self.W.double(), self.W.T.double())
        self.summation2 += -torch.trace(a2)
        
        b1 = torch.matmul(self.W.double(), Y_hat.reshape(1, -1).double())
        b2 = torch.matmul(b1.T.reshape(-1, 2, 1), S.reshape(-1, 1, 2).double())
        b3 = torch.matmul(b2, self.P_s_sqrt_inv.repeat(current_batch_size, 1).reshape(current_batch_size, 2, 2).double().to(self.device))
        
        # 3dim tensor trace
        mask = torch.zeros((current_batch_size, 2, 2))
        mask[:, torch.arange(0,2), torch.arange(0,2)] = 1.0
        b4 = b3 * mask.to(self.device)
        b5 = torch.sum(b4)
        b6 = (2 * b5) - (1 * current_batch_size)

        self.summation2 += b6
        
        return self.lam * (self.summation2 / current_batch_size)