from eval import calculate_evaluation_metrics, print_final_results
import time


csv_path = "eval_results/predictions_20250807_114826.csv"
output_path = "eval_results"
time_stamp = time.strftime('%Y%m%d_%H%M%S')

metric = calculate_evaluation_metrics(csv_path, output_path, time_stamp)
print_final_results(metric)