import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compare_models(results):
    """
    Compare model performance across multiple metrics.

    Args:
        results (dict): Dictionary containing model evaluation metrics.
    """

    logging.info("Plotting graph for comparison")

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(results.keys())
    num_metrics = len(metrics)

    # Create a grouped bar chart
    x = np.arange(num_metrics)
    width = 0.2

    plt.figure(figsize=(8, 6))

    for i, model in enumerate(model_names):
        scores = [results[model][metric] for metric in metrics]
        plt.bar(x + i * width, scores, width=width, label=model)

    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Model Performance Comparison")
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.ylim(0, 1)  # Scores range from 0 to 1

    plt.show()

