library("tidyverse")
library("ggplot2")
library("dplyr")
library("summarytools")
library("skimr")
library("rsample")
library("recipes")
library("parsnip")
library("tidymodels")
library("yardstick")
library("pROC")
library("ranger")
library("randomForest")
library("caret")
library("kknn")
loan_data <- readRDS("loan_data.rds")

#Exploring the dataset
dim(loan_data) 
str(loan_data) 
head(loan_data) 
glimpse(loan_data)

summary(loan_data)
skim(loan_data)

#The dataset is already clean and does not contain any missing values so we can being Exploring the dataset now.

df_summary <- loan_data %>%
  group_by(loan_default) %>%
  summarise(count = n())

ggplot(df_summary, aes(x = loan_default, y = count, fill = loan_default)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Count of Loan Default",
       y = "Count",
       x = "Loan Default") +
  theme_minimal()

# Calculating the percentage
df_summary$percentage <- (df_summary$count / sum(df_summary$count)) * 100
print(df_summary)

#The positive loan default rate for customers is quite considering that it is 37.2% 
#Lets take a look if we can find strong correlations between other factors in our dataset to understand why this might be the case.


#Question 1 :  What purpose of the loan has the highest rate of defaulting ?
loan_data %>% 
  group_by(loan_default,loan_purpose)%>%
  summarize(no_of_defaults=n())

loan_data %>%
  group_by(loan_purpose) %>%
  summarize(default_rate = mean(loan_default == "yes")) %>%
  ggplot(aes(x = reorder(loan_purpose, -default_rate), y = default_rate)) +
  geom_bar(stat = "identity", fill = "purple") +
  labs(title = "Default Rate by Loan Purpose", x = "Loan Purpose", y = "Default Rate") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#We observe that Medical loans and credit card loans have the highest rate of defaults.
#Where medical loan's default rate is at 60.5% and credit card loan's default rate is at 53.5%


#Question 2 : Is there a difference in default rate between loans of different terms?

#Calculating the default rate for each loan term
default_loanterm <- loan_data %>%
  group_by(term) %>%
  summarize(default_rate = mean(loan_default == "yes"))

default_loanterm

#There are two types of loan term which are 3years and 5years.
#what we observe that the default rate on a 5year loan is quite high compared to the 3year loan.


#visualization:
default_loanterm %>%
  ggplot(aes(x = term, y = default_rate)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "Default Rate by Loan Term", x = "Loan Term", y = "Default Rate") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

#We can observe that a three year loan has a default rate of 26.8 % where as a loan period of 5 years has a default rate of 55%


#Question 3: Did the applicants with a higher interest rate have a higher rate of default?

loan_data$Interest_Rate_Group <- cut(loan_data$interest_rate, 
                                     breaks = c(0, 10, 15, 20), 
                                     labels = c("0-10%", "10-15%", "15-20%"))

table_default_interest <- table(loan_data$Interest_Rate_Group, loan_data$loan_default)

# Calculate default rates within each interest rate group
default_rates <- prop.table(table_default_interest, margin = 1) * 100  # Convert to percentages
default_rates

ggplot(data = loan_data, aes(x = Interest_Rate_Group, fill = loan_default)) +
  geom_bar(position = "stack") +
  labs(title ="Default and Non-Default Cases by Interest Rate Group",
       x = "Interest Rate Group",
       y = "Count") +
  theme_minimal()

loan_data%>%
  group_by(loan_default)%>%
  summarize(defaults = n(),avg_interest_rate=mean(interest_rate))

#Question 4: Did applicants with a lower average annual income have a higher rate of default?
income_summary <- loan_data%>%
  group_by(loan_default)%>%
  summarise(mean_annual_income = mean(annual_income),Count=n())

print(income_summary)

income_plot <- ggplot(income_summary, aes(x = loan_default, y = mean_annual_income, fill = loan_default)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Mean Annual Income by Loan Default Status",
       x = "Loan Default Status",
       y = "Mean Annual Income") +
  scale_fill_manual(values = c("yes" = "red", "no" = "steelblue")) +  # Customize colors
  theme_minimal()

income_plot

#Question 5: Is there a relationship between defaulting on the loan, loan purpose and loan amount?
loan_sum_5 <- loan_data %>%
  group_by(loan_default, loan_purpose) %>%
  summarise(mean_loan_amount = mean(loan_amount), Count = n())

loan_sum_5

loan_plot <- ggplot(loan_sum_5, aes(x = loan_purpose, y = mean_loan_amount, fill = loan_default)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Mean Loan Amount by Loan Purpose and Default Status",
       x = "Loan Purpose",
       y = "Mean Loan Amount") +
  scale_fill_manual(values = c("yes" = "red", "no" = "steelblue")) +  # Customize colors
  theme_minimal()

loan_plot

#Predictive Modeling:
#Initializing split and feature selection 

set.seed(123)
loans_split <- initial_split(loan_data , prop = 0.80, strata = loan_default )
loan_train <- loans_split %>% training()
loan_test <- loans_split %>% testing()
set.seed(123)

#Cross validation fold of 10 is selected
loans_folds <- vfold_cv(loan_train, v = 10)

#Defining a recipe for preprocessing the data
loans_rec <- recipe(loan_default ~., data = loan_train) %>%
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

#Training data is preprocessed
loans_rec%>%
  prep() %>%
  bake(new_data = loan_train)

#First model is a logistic regression model:
loan_log_model <- logistic_reg() %>%
  set_engine('glm') %>%
  set_mode('classification')

loan_log_model

#workflow and model fit
log_pip <- workflow()%>%
  add_model(loan_log_model)%>%
  add_recipe(loans_rec)

model_fit <- log_pip%>%
  last_fit(split=loans_split)

#model prediction stored in model 1
model1 <- model_fit %>%
  collect_predictions()

str(model1)

#ROC AUC 

roc_auc(model1, truth = loan_default, .pred_yes)

conf_mat(model1, truth = loan_default, estimate = .pred_class)



#Applying KNN method as the second model for classification:
#Creating the knn model while hypertuning
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')
knn_model

#Making a workflow with the recipe defined above
knn_pip <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(loans_rec)
k_grid <- tibble(neighbors = c(10, 15, 25, 45, 60, 80, 100, 120, 140, 180))
#grid parameter is defined which is number of 'k' to tune over
k_grid

#Applying cross validation with k=10 
set.seed(231)
knn_tuning <- knn_pip %>%
  tune_grid(resamples = loans_folds, grid = k_grid)
best_k <- knn_tuning %>%
  select_best(metric = 'roc_auc')


#Best k is selected
best_k

#Model fit
final_knn <- knn_pip %>%
  finalize_workflow(best_k)

knn_model <- final_knn %>%
  last_fit(split = loans_split)

#model prediction stored in 'model2'
model2 <- knn_model %>%
  collect_predictions()

model2

#ROC AUC 
roc_auc(model2, truth = loan_default, .pred_yes)

conf_mat(model2, truth = loan_default, estimate = .pred_class)


