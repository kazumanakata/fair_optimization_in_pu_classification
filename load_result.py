from matplotlib import pyplot as plt
import numpy as np

list1 = np.load('experiment_result/train_risk_list.npy')
list2 = np.load('experiment_result/train_fair_list.npy')
list3 = np.load('experiment_result/test_loss_list.npy')
list4 = np.load('experiment_result/train_acc_list.npy')
list5 = np.load('experiment_result/test_acc_list.npy')
list6 = np.load('experiment_result/test_P_acc_list.npy')
list7 = np.load('experiment_result/test_N_acc_list.npy')
list8 = np.load('experiment_result/test_dp_list.npy')


def show_fig(list1):
    fig = plt.figure()
    for i in range(len(list1)):
        plt.plot(np.arange(list1.shape[1]), list1[i])
    plt.legend()
    plt.show()

show_fig(list1)
show_fig(list2)
show_fig(list3)
show_fig(list4)
show_fig(list5)
show_fig(list6)
show_fig(list7)
show_fig(list8)
