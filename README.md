# fair_optimization_in_pu_classification

This is a repository for implementing a stochastic optimization framework for fair risk minimization in PU classification.

## Fairness in Machine Learning
In Machine learning applications, it is crucial to  eliminate unequal outcomes for certain groups that possess some sensitive attributes. For instance, in housing loans, predictors judging applicants' repayment capabilities should not consider their gender or race. To make sure classification results are uncorrelated to such attributes, we can formulate a minimization framework for training predictors. In general, ERM (Empirical Risk Minimization) [1] is used to obtain a predictor that minimizes the empirical risk for training data. In fair risk minimization, we introduce a regularization term that incorporates mutual information. This repository makes use of the ERMI (exponential Rényi mutual information) [2] between the predictors' outputs and sensitive attributes of training data. By adding this term to ERM, the framework returns the predictor that make outputs uncorrelated to the sensitive attributes.

## PU classification
In supervised machine learning, collecting an enormous amount of labeled data for training predictors can be laborious. To reduce the effort of labeling, PU (Positive and Unlabeled) classification [3] aims to obtain a predictor only from positive and unlabeled data. PU classification is one of the weakly supervised learning and its consistency is guaranteed (Its unbiasedness is also guaranteed for the Unbiased PU classification).

## Fairness in PU classification
Employing PU classification in real world scenarios may lead to a higher possibilities resulting in unfair predictors compared to ordinary PN (Positive and Negative) classification. For example, in PU classification, the framework identifies training samples likely to be negative class  based on positively labeled samples. If some groups are labeled for positive class based on sensitive attributes, other groups might naively be considered as negative class based on the same sensitive attributes. To prevent predictors from being unfair in PU classification, we consolidate a fairness regularizer term into PU classification framework.

## Codes
- Please unzip `mnist_train.zip` before running the code (Compressed due to its file size).
- `fermi.py` has a torch implementation of the regularizer for fair risk minimization.
- `execute.py` is a code for training a fair predictor in PU classification. MNIST [4] are used in a way that even numbers are labeled as positive class and odd numbers are labeled as negative class. Also a value one is encoded for images of {2, 5, 8} and a value zero is encoded for images of {0, 1, 3, 4, 6, 7, 9} as binary sensitive attributes, mixing positive and negative classes. In a PU classification setting, by default 1000 positive samples and 59000 unlabeled samples are used for training a predictor. To investigate the effectiveness of fairness regularizer, the coefficient *lambda* for it is changed to values from {0, 1, 2, 3, 4, 5}. Here, *lambda* decides how much the predictor considers the regularization term in training.

You can run the code by
```
python3 main.py True 200
```
Here, the first argument specifies if it is PU scenario or not and the second argument specifies the number of epochs for training. You can see additional information by adding `--help`. 


## Requirements
- Python == 3.9.13
- Numpy == 1.21.5
- Torch == 1.13.1
- Fairlearn == 0.8.0
- Matplotlib == 3.5.2
- Pandas == 1.4.4

## Example result
***Below, you can see that as the lambda increases, the predictor tries to be fair for sensitive attributes. As a result, it tends to output negative class regardless of input samples' features.***

After running `main.py`, 8 figures are shown and `.npz` files are stored in `experiment_result/`.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/e64ae5a2-f502-4b71-b192-8416bf8ef5a2"><br>
This is a graph showing empirical loss.<br>
As the *lambda* takes higher values, the predictor would become less accurate for classifying training data and empirical loss takes higher values.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/af6a6029-ad01-453b-8cc4-3df788f826c2"><br>
This is a graph showing regularization loss.<br>
As the *lambda* takes higher values, the predictor would become more fair for sensitive attributes and regularization loss takes lower values.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/d9b3157f-870b-49fd-8f98-77f2e3f409b2"><br>
This is a graph showing expected loss.<br>
As the *lambda* takes higher values, the predictor would become less accurate for classifying test data and expected loss takes higher values.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/09e4f30a-1d0f-4636-a65a-1b4ea2207214"><br>
This is a graph showing accuracy for training data.<br>
As the *lambda* takes higher values, the predictors' accuracy for training data increases.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/21ee84a6-b891-45b4-b99a-4e26650d6602"><br>
This is a graph showing accuracy for test data.<br>
In contrast to the accuracy for training data, as the *lambda* takes higher values, the predictors' accuracy for test data decreases.<br>
This indicates the trained predictor tends to output negative labels because it aims to be fair for sensitive attributes {1, 0}.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/5195e0fa-ea89-431f-a3a5-c62fc9908d87"><br>
This is a graph showing accuracy for positive class in test data.<br>


<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/b0712289-b7d3-47bd-a89a-1fc9dd8efb05"><br>
This is a graph showing accuracy for negative class in test data.

<img src="https://github.com/kazumanakata/fair_optimization_in_pu_classification/assets/121463877/9c72fa7d-846e-43e3-891c-2b73944bb5fb"><br>
This is a graph showing demographic parity for test data.

## Reference
1. Mehryar Mohri Afshin Rostamizadeh, and Ameet Talwalkar: Foundations of Machine Learning., MIT Press, Second Edition, 2018.
1. Andrew Lowy, Sina Baharlouei, Rakesh Pavan, Meisam Razaviyayn, and Ahmad Beirami: A Stochastic Optimization Framework for Fair Risk Minimization., Transactions on Machine Learning Research, 2022.
1. Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, and Masashi Sugiyama: Positive-Unlabeled Learning with Non-Negative Risk Estimator, 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, US.
1. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition., Proceedings of the IEEE, 86(11):2278-2324, November 1998.
