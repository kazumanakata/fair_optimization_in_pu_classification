import numpy as np
import torch
from torch.utils.data import DataLoader
from fermi import FERMI
from mnist_dataset import MNIST
from train import train_test
from matplotlib import pyplot as plt

def plot_graph(lambda_list, epochs, result_list):
    
    fig = plt.figure()
    
    for i, lam in enumerate(lambda_list):
        result = result_list[i]
        plt.plot(np.arange(epochs), result, label='lam:'+str(lam))
        
    plt.legend()
    plt.show()


def executer(pu_train, epochs, p_num, device, X_train, X_test, y_train, y_test, y_train2, y_test2):
     
    lambda_list = list(range(6))
    seed_list = list(range(5))
    num_P_in_PU = p_num
    
    train_risk_list = np.zeros((len(lambda_list), epochs))
    train_fair_list = np.zeros((len(lambda_list), epochs))
    test_loss_list  = np.zeros((len(lambda_list), epochs))
    train_acc_list  = np.zeros((len(lambda_list), epochs))
    test_acc_list   = np.zeros((len(lambda_list), epochs))
    test_P_acc_list = np.zeros((len(lambda_list), epochs))
    test_N_acc_list = np.zeros((len(lambda_list), epochs))
    test_dp_list    = np.zeros((len(lambda_list), epochs))
    

    for l, lam in enumerate(lambda_list):
        lr_W = 1.0
    
        train_risk_arr = np.zeros((len(seed_list), epochs))
        train_fair_arr = np.zeros((len(seed_list), epochs))
        test_loss_arr  = np.zeros((len(seed_list), epochs))
        train_acc_arr  = np.zeros((len(seed_list), epochs))
        test_acc_arr   = np.zeros((len(seed_list), epochs))
        test_P_acc_arr = np.zeros((len(seed_list), epochs))
        test_N_acc_arr = np.zeros((len(seed_list), epochs))
        test_dp_arr    = np.zeros((len(seed_list), epochs))
    
        for seed in seed_list:
            print('lambda:', lam, 'seed:', seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
    
            new_idx_train = np.random.permutation(np.arange(len(X_train)))
            new_idx_test = np.random.permutation(np.arange(len(X_test)))
    
            X_train_sample  = X_train.copy()[new_idx_train]
            X_test_sample   = X_test.copy()[new_idx_test]
            y_train_sample  = y_train.copy()[new_idx_train]  # 1,0  label
            y_test_sample   = y_test.copy()[new_idx_test]    # 1,0  label
            y_train2_sample = y_train2.copy()[new_idx_train] # 1,-1 label
            y_test2_sample  = y_test2.copy()[new_idx_test]   # 1,-1 label
    
            if pu_train:
                cnt1 = num_P_in_PU
                cnt_P_in_U = 0
                for i in range(len(y_train2_sample)):
                    if (y_train2_sample[i] == 1) and (cnt1 > 0):
                        y_train2_sample[i] = 1
                        cnt1 -= 1
                    elif (y_train2_sample[i] == 1) and (cnt1 <= 0):
                        y_train2_sample[i] = -1
                        cnt_P_in_U += 1
    
                print("num P:", np.sum(y_train2_sample==1), "num U:", np.sum(y_train2_sample==-1))
                print("ratio of P in Unlabeled data:", cnt_P_in_U / np.sum(y_train2_sample==-1))
    
            # 3rd feature from {1, 0}
            therd_feat_train = []
            for row in range(len(y_train_sample)):
                if y_train_sample[row] in [2,5,8]:
                    therd_feat_train.append(1)
                elif y_train_sample[row] in [0,1,3,4,6,7,9]:
                    therd_feat_train.append(0)
            therd_feat_train = np.array(therd_feat_train).reshape(-1,1)
    
            therd_feat_test = []
            for row in range(len(y_test_sample)):
                if y_test_sample[row] in [2,5,8]:
                    therd_feat_test.append(1)
                elif y_test_sample[row] in [0,1,3,4,6,7,9]:
                    therd_feat_test.append(0)
            therd_feat_test = np.array(therd_feat_test).reshape(-1,1)
    
            # make one-hot encoding s_i
            S_train = np.zeros((len(therd_feat_train), 2))
            for i in range(len(therd_feat_train)):
                if therd_feat_train[i] == 1:
                    S_train[i,1] += 1
                else:
                    S_train[i,0] += 1
    
            S_test = np.zeros((len(therd_feat_test), 2))
            for i in range(len(therd_feat_test)):
                if therd_feat_test[i] == 1:
                    S_test[i,1] += 1
                else:
                    S_test[i,0] += 1
    
            p_hat_s01 = np.sum(S_train[:,1]) / len(S_train)
            p_hat_s10 = np.sum(S_train[:,0]) / len(S_train)
            p_hat_s = np.array([[p_hat_s01,0],[0,p_hat_s10]])
    
            print(p_hat_s)
    
            # Make datasets
            train_dataset = MNIST(X_train_sample, y_train2_sample, si=S_train)
            test_dataset = MNIST(X_test_sample, y_test2_sample, si=S_test)
            print("train_dataset size:", train_dataset.__len__(), "test_dataset size:", test_dataset.__len__())
    
            # Make data loaders
            batch_size=30000
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
            model = FERMI(X_train, X_test, y_train.reshape(-1,1), y_test, S_train, S_test, device, lam=lam)
            mean_train_risk_losses, mean_train_fair_losses, mean_test_losses, mean_train_accuracy, mean_test_accuracy, \
                                mean_test_accuracy_sens1, mean_test_accuracy_sens0, mean_test_P_accuracy, mean_test_N_accuracy, \
                                mean_test_dp = train_test(epochs, model, train_loader, test_loader, pu_train, lr_W, device)
    
            train_risk_arr[seed,:] += np.array(mean_train_risk_losses)
            train_fair_arr[seed,:] += np.array(mean_train_fair_losses)
            test_loss_arr[seed,:]  += np.array(mean_test_losses)
            train_acc_arr[seed,:]  += np.array(mean_train_accuracy)
            test_acc_arr[seed,:]   += np.array(mean_test_accuracy)
            test_P_acc_arr[seed,:] += np.array(mean_test_P_accuracy)
            test_N_acc_arr[seed,:] += np.array(mean_test_N_accuracy)
            test_dp_arr[seed,:]    += np.array(mean_test_dp)
    
        train_risk_list[l,:] = np.sum(train_risk_arr, axis=0) / len(seed_list)
        train_fair_list[l,:] = np.sum(train_fair_arr, axis=0) / len(seed_list)
        test_loss_list[l,:]  = np.sum(test_loss_arr,  axis=0) / len(seed_list)
        train_acc_list[l,:]  = np.sum(train_acc_arr,  axis=0) / len(seed_list)
        test_acc_list[l,:]   = np.sum(test_acc_arr,   axis=0) / len(seed_list)
        test_P_acc_list[l,:] = np.sum(test_P_acc_arr, axis=0) / len(seed_list)
        test_N_acc_list[l,:] = np.sum(test_N_acc_arr, axis=0) / len(seed_list)
        test_dp_list[l,:]    = np.sum(test_dp_arr,    axis=0) / len(seed_list)
    
    np.save('experiment_result/train_risk_list', train_risk_list)
    np.save('experiment_result/train_fair_list', train_fair_list)
    np.save('experiment_result/test_loss_list', test_loss_list)
    np.save('experiment_result/train_acc_list', train_acc_list)
    np.save('experiment_result/test_acc_list', test_acc_list)
    np.save('experiment_result/test_P_acc_list', test_P_acc_list)
    np.save('experiment_result/test_N_acc_list', test_N_acc_list)
    np.save('experiment_result/test_dp_list', test_dp_list)
    
    
    plot_graph(lambda_list, epochs, train_risk_list)
    plot_graph(lambda_list, epochs, train_fair_list)
    plot_graph(lambda_list, epochs, test_loss_list)
    plot_graph(lambda_list, epochs, train_acc_list)
    plot_graph(lambda_list, epochs, test_acc_list)
    plot_graph(lambda_list, epochs, test_P_acc_list)
    plot_graph(lambda_list, epochs, test_N_acc_list)
    plot_graph(lambda_list, epochs, test_dp_list)