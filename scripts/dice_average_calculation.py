
import pandas as pd

# Load the results CSV file
test_results_dir_path = "/data/test_results"
csv_results_filename = test_results_dir_path + "/results_test_trained_on_gentuity"

results_df_fold0 = pd.read_csv(f"{csv_results_filename}fold0.csv")
results_df_fold1 = pd.read_csv(f"{csv_results_filename}fold1.csv")
results_df_fold2 = pd.read_csv(f"{csv_results_filename}fold2.csv")
results_df_fold3 = pd.read_csv(f"{csv_results_filename}fold3.csv")
results_df_fold4 = pd.read_csv(f"{csv_results_filename}fold4.csv")
average_dice_scores_fold0 = results_df_fold0.groupby("Model")["Dice Score"].mean()
average_dice_scores_fold1 = results_df_fold1.groupby("Model")["Dice Score"].mean()
average_dice_scores_fold2 = results_df_fold2.groupby("Model")["Dice Score"].mean()
average_dice_scores_fold3 = results_df_fold3.groupby("Model")["Dice Score"].mean()
average_dice_scores_fold4 = results_df_fold4.groupby("Model")["Dice Score"].mean()

# Calculate average Dice scores across all folds for each model
average_dice_scores_all_folds = pd.concat([
    average_dice_scores_fold0,
    average_dice_scores_fold1,
    average_dice_scores_fold2,
    average_dice_scores_fold3,
    average_dice_scores_fold4
], axis=1).mean(axis=1)

# Print the average Dice scores for each model across all folds
print("Average Dice scores across all folds:")
print(average_dice_scores_all_folds)