library(tidymodels)
library(vroom)

# Load Data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

# Ensure 'ACTION' is a Factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Recipe for Pre-processing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

# Model Specification
my_mod <- logistic_reg() %>%
  set_engine("glm")

# Workflow
amazon_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(my_mod)

# Split Data for Validation
set.seed(123)
data_split <- initial_split(amazonTrain, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Fit and Evaluate Model
fit_workflow <- fit(amazon_workflow, data = train_data)

# Predict Probabilities for Validation
predictions <- predict(fit_workflow, test_data, type = "prob") %>%
  bind_cols(test_data)

# Evaluate Metrics
metrics <- metric_set(roc_auc, accuracy, precision, recall)
eval_results <- metrics(predictions, truth = ACTION, estimate = .pred_1)
print(eval_results)

# Predict Probabilities for Test Data
amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "prob")

# Create Submission Data Frame

amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "class")

submission_df <- data.frame(Id = seq_len(nrow(amazonTest)), Action = amazon_predictions$.pred_class)
vroom_write(x = submission_df, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/submission2.csv", delim = ",")
