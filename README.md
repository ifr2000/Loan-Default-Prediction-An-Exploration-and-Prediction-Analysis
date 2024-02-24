# Loan-Default-Prediction-An-Exploration-and-Prediction-Analysis


## Introduction

This project delves into the analysis of loan default data from over 3,500 individuals who secured personal loans in 2017 from a national bank. The core objective is to uncover the factors leading to loan defaults and to construct a machine learning model capable of predicting the likelihood of future defaults. This initiative aims to equip the bank with insights to mitigate financial losses stemming from loan defaults, a pressing issue highlighted by recent record default levels.

## Dataset Overview

The dataset encompasses a range of factors including loan amounts, interest rates, purposes of loans, and applicant details. With an already clean dataset, our exploration dives directly into an in-depth EDA (Exploratory Data Analysis) to address five pivotal research questions, shedding light on the characteristics and behaviors influencing loan defaults.

## Key Findings

1. **Loan Purpose and Default Rates**: Credit card loans exhibit the highest default rate at 60.5%, followed by medical and debt consolidation loans. This insight suggests a significant risk associated with credit card loans.
2. **Impact of Loan Terms**: Loans with a five-year term show a default rate of 55%, starkly higher than the three-year term loans at 26.8%. The duration of a loan emerges as a critical factor in default risks.
3. **Interest Rates Correlation**: A direct correlation between higher interest rates and increased default rates is evident, with a 100% default rate for loans carrying 15-20% interest rates.
4. **Annual Income Influence**: Borrowers who defaulted had a lower average annual income compared to those who didn't, indicating financial stability as a determinant in loan repayment capabilities.
5. **Loan Amount and Defaulting**: Higher loan amounts, especially for debt consolidation and small business purposes, are associated with increased default rates.

## Predictive Modeling

Two classification models, Logistic Regression and K-Nearest Neighbors (KNN), were developed to predict loan defaults. The Logistic Regression model demonstrated superior performance with a 95.28% accuracy rate, significantly outperforming the KNN model.

## Conclusion and Recommendations

The analysis underscores the critical impact of loan terms, interest rates, and financial stability on default rates. Recommendations for the bank include:

- Prioritizing the issuance of loans with interest rates below 10% to minimize default risks.
- Reevaluating the strategy around five-year loans due to their high default rates.
- Employing predictive models, particularly the Logistic Regression model, for assessing loan default risks.

By implementing these strategies, the bank can enhance its loan issuing policies to mitigate default risks effectively.

