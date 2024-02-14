import sys
import argparse
import numpy as np
import pandas as pd
import torch
from execute import executer

def main(arguments):
    parser = argparse.ArgumentParser(description='An implementation of a stochastic optimization framework for fair risk minimization in PU classification')
    
    parser.add_argument('pu_train', default=True, type=bool, help='True: PU classification, False: PN classification')
    parser.add_argument('epochs', default=200, type=int, help='# of training epochs')
    parser.add_argument('--p_num', default=1000, type=int, help='# of positively labeled samples in training data')
    
    args = parser.parse_args(arguments)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    train = pd.read_csv("mnist_train.csv")
    test = pd.read_csv("mnist_test.csv")
    
    train_np = train.values
    test_np = test.values
    
    X_train = train_np[:,1:]
    X_test  = test_np[:,1:]
    y_train = train_np[:,0]
    y_test  = test_np[:,0]
    
    y_train2 = np.where(y_train%2 == 0, 1, -1)
    y_test2  = np.where(y_test %2 == 0, 1, -1)
    
    print('X_train.shape:', X_train.shape, 'X_test.shape:', X_test.shape)
    print('train pos label num:', np.sum(y_train2==1), 'train neg label num:', np.sum(y_train2==-1), 'test pos label num:', np.sum(y_test2==1), 'test neg label num:', np.sum(y_test2==-1))
    
    executer(args.pu_train, args.epochs, args.p_num, device, X_train, X_test, y_train, y_test, y_train2, y_test2)


if __name__ == '__main__':
    main(sys.argv[1:])