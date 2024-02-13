# fair_optimization_in_pu_classification

This is a repository for implementing a stochastic optimization framework for fair risk minimization in PU classificaiton.

## fairness in machine learning
In applications of machine learning, sometimes we want to eleminate inequalized outputs for certain groups that have some sensitive attributes. For instance, when it comes to housing loan, predictors judging applicants have capability of repayment should not account for their genders or races. To make classification results uncorrelated to such atttributes, we can formulate minimization framework for training predictors. In general, ERM (Empirical Risk Minimization) is used to obtain a predictor which minimizes the empirical risk for training data. In fair risk minimization, we can add a regularization term which incorporates mutual information. In this repository, we address the ERMI (exponential Rényi mutual information) between the predictors' outputs and sensitive attributes of training data. As we add this term for ERM, the framework returns the predictor which would make outputs uncorrelated to the senstive attributes.

## PU classification
In supervised machine learning, we have to collect an enormous amount of labels for training predictors. In order to mitigate the labor of labeling, PU (Positive and Unlabeled) classification aims to obtain a predictor only from positive and unlabeled data. PU classification is one of the weakly supervised learning and its consistency is guranteed (Its unbiasedness is also guranteed for the Unbiased PU classification).

## fairness in PU classification
When we employ PU classification for real world, there might be more possibities resulting in unfair predictors than ordinary PN (Positive and Negative) classification. For example, in PU classification, the framework seeks training samples which likely have negative labels based on the positively labeled samples. If some groups have been labeled for positive class based on the sensitive attributes, some other groups would naively be considered as negative class based on the sensitive attributes. To prevent predictors from being unfair in PU classification, we consolidate fairness reguralizer term into PU classification.
