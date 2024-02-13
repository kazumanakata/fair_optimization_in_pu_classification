# fair_optimization_in_pu_classification

This is a repository for implementing a stochastic optimization framework for fair risk minimization in PU classificaiton.

## Fairness in Machine Learning
In Machine learning applications, it is crucial to  eliminate unequal outcomes for certain groups that possess some sensitive attributes. For instance, in housing loans, predictors judging applicants' repayment capabilities should not consider their gender or race. To make sure classification results are uncorrelated to such atttributes, we can formulate a minimization framework for training predictors. In general, ERM (Empirical Risk Minimization) [1] is used to obtain a predictor that minimizes the empirical risk for training data. In fair risk minimization, we introduce a regularization term that incorporates mutual information. This repository makes use of the ERMI (exponential Rényi mutual information) [2] between the predictors' outputs and sensitive attributes of training data. By adding this term to ERM, the framework returns the predictor that make outputs uncorrelated to the senstive attributes.

## PU classification
In supervised machine learning, collecting an enormous amount of labeled data for training predictors can be laborious. To reduce the effort of labeling, PU (Positive and Unlabeled) classification [3] aims to obtain a predictor only from positive and unlabeled data. PU classification is one of the weakly supervised learning and its consistency is guranteed (Its unbiasedness is also guranteed for the Unbiased PU classification).

## Fairness in PU classification
Employing PU classification in real world scenarios may lead to a higher possibities resulting in unfair predictors compared to ordinary PN (Positive and Negative) classification. For example, in PU classification, the framework identifies training samples likely to be negative class  based on positively labeled samples. If some groups are labeled for positive class based on sensitive attributes, other groups might naively be considered as negative class based on the same sensitive attributes. To prevent predictors from being unfair in PU classification, we consolidate a fairness reguralizer term into PU classification framework.

## Reference
1. Mehryar Mohri Afshin Rostamizadeh, and Ameet Talwalkar: Foundations of Machine Learning., MIT Press, Second Edition, 2018.
1. Andrew Lowy, Sina Baharlouei, Rakesh Pavan, Meisam Razaviyayn, and Ahmad Beirami: A Stochastic Optimization Framework for Fair Risk Minimization., Transactions on Machine Learning Research, 2022
1. Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, and Masashi Sugiyama: Positive-Unlabeled Learning with Non-Negative Risk Estimator, 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, US
