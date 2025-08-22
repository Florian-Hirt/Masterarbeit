import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import os
import json
import numpy as np


def analyze_score_distributions(logger, score_distributions: Dict[str, List[int]], output_dir: str = "") -> None:
    filtered_distributions: Dict[str, List[int]] = {}
    for sample_id, scores in score_distributions.items():
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            filtered_distributions[sample_id] = valid_scores
    
    all_scores = [score for scores in filtered_distributions.values() for score in scores]
    
    if not all_scores:
        logger.warning("No valid scores to analyze for overall distribution.")
        return
    
    score_counts = Counter(all_scores)
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    variance = np.var(all_scores)
    logger.info(f"Overall statistics - Mean: {mean_score:.2f}, Median: {median_score:.2f}, Variance: {variance:.2f}")
    
    plt.figure(figsize=(12, 10)) # Adjusted figure size
    
    plt.subplot(2, 1, 1)
    plt.hist(all_scores, bins=np.arange(0.5, 6.5, 1), alpha=0.7, edgecolor='black')
    plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='g', linestyle='--', label=f'Median: {median_score:.2f}')
    plt.title(f'Distribution of Adversarial Scores (Valid Evaluations Only)\nVariance: {variance:.2f}')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    x_labels = sorted(score_counts.keys())
    y_values = [score_counts[score] for score in x_labels]
    plt.bar(x_labels, y_values, color='skyblue', edgecolor='black')
    plt.title('Count of Each Score Value (Valid Evaluations Only)')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    for i, count in enumerate(y_values):
        plt.text(x_labels[i], count + 0.05 * max(y_values), str(count), ha='center') # Relative offset for text
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'score_distribution_summary.png')
    plt.savefig(fig_path)
    logger.info(f"Score distribution summary visualization saved to {fig_path}")
    plt.close() # Close the figure
    
    sample_stats: Dict[str, Dict[str, Union[int, float, str]]] = {} # Adjusted type for 'id'
    for sample_id_key, scores in filtered_distributions.items():
        sample_stats[str(sample_id_key)] = { # Ensure sample_id is string for JSON
            'min': min(scores), 'max': max(scores),
            'mean': np.mean(scores), 'median': np.median(scores),
            'variance': np.var(scores), 'count': len(scores),
            'valid_percentage': (len(scores) / len(score_distributions.get(sample_id_key, []))) * 100
                               if score_distributions.get(sample_id_key) else 0
        }
    
    stats_path = os.path.join(output_dir, 'per_sample_score_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(sample_stats, f, indent=2)
    logger.info(f"Per-sample score statistics saved to {stats_path}")

def generate_per_sample_histograms(
    logger,
    score_distributions: Dict[str, List[int]], 
    base_output_dir: str = "",
    histogram_dir: str = "sample_histograms",
    original_scores: Optional[Dict[str, int]] = None # Make original_scores optional
) -> None:
    output_dir_path = os.path.join(base_output_dir, histogram_dir) # Renamed to avoid conflict
    os.makedirs(output_dir_path, exist_ok=True)
    
    total_samples = len(score_distributions)
    logger.info(f"Generating histograms for {total_samples} samples into {output_dir_path}")
    
    for i, (sample_id, scores) in enumerate(score_distributions.items()):
        valid_scores = [score for score in scores if score is not None]
        if not valid_scores:
            logger.warning(f"Sample {sample_id} has no valid scores, skipping histogram.")
            continue
        
        mean_score = np.mean(valid_scores)
        median_score = np.median(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        score_variance = np.var(valid_scores)
        
        original_score_val = None # Renamed
        if original_scores and sample_id in original_scores:
            original_score_val = original_scores[sample_id]
        
        str_id = str(sample_id)
        clean_id = ''.join(c for c in str_id if c.isalnum() or c in '._-')
        
        plt.figure(figsize=(10, 6))
        score_counts = Counter(valid_scores)
        bins = np.arange(0.5, 6.5, 1)
        plt.hist(valid_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
        
        plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.2f}')
        
        if original_score_val is not None:
            plt.axvline(original_score_val, color='purple', linestyle='-', linewidth=3, label=f'Original Score: {original_score_val}')
            max_height = plt.gca().get_ylim()[1]
            marker_height = max_height * 0.75
            plt.plot([original_score_val], [marker_height], 'v', color='purple', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
            plt.text(original_score_val, marker_height * 1.05, f"Original: {original_score_val}", color='purple', fontweight='bold', ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='purple', boxstyle='round,pad=0.2'))
        
        for score_val_hist in range(1, 6): # Renamed
            count = score_counts.get(score_val_hist, 0)
            if count > 0:
                plt.text(score_val_hist, count + 0.02 * max(score_counts.values(), default=1), str(count), ha='center', fontweight='bold') # Adjusted offset

        title = f'Rating Distribution for Sample {str_id}\n'
        if original_score_val is not None: title += f'Original Score: {original_score_val}, '
        title += f'Min: {min_score}, Max: {max_score}, Var: {score_variance:.2f}, n={len(valid_scores)}'
        
        plt.title(title, fontsize=12)
        plt.xlabel('Rating (1-5)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.xticks(range(1, 6))
        plt.grid(True, alpha=0.4, linestyle=':')
        plt.legend()
        
        output_path_fig = os.path.join(output_dir_path, f'histogram_sample_{clean_id}.png') # Renamed
        plt.tight_layout()
        plt.savefig(output_path_fig, dpi=100)
        plt.close()
        
        if (i + 1) % 10 == 0 or i + 1 == total_samples:
            logger.info(f"Generated {i + 1}/{total_samples} per-sample histograms.")
    
    logger.info(f"All per-sample histograms saved to {output_dir_path}/")

def generate_histogram_grid(
    logger,
    score_distributions: Dict[str, List[int]],
    max_samples: int = 16,
    output_dir: str = "",
    output_filename: str = "histogram_grid.png",
    original_scores: Optional[Dict[str, int]] = None
) -> None:
    output_path = os.path.join(output_dir, output_filename)
    valid_samples = {}
    for sample_id, scores in score_distributions.items():
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            valid_samples[sample_id] = valid_scores
            if len(valid_samples) >= max_samples:
                break
    if not valid_samples:
        logger.warning("No valid samples with scores, cannot generate histogram grid.")
        return

    n_samples = len(valid_samples)
    grid_size_cols = int(np.ceil(np.sqrt(n_samples)))
    grid_size_rows = int(np.ceil(n_samples / grid_size_cols))

    fig, axes = plt.subplots(grid_size_rows, grid_size_cols, figsize=(grid_size_cols * 4, grid_size_rows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    for i, (sample_id, scores) in enumerate(valid_samples.items()):
        ax = axes_flat[i]
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        original_score_val = original_scores.get(sample_id) if original_scores else None

        ax.hist(scores, bins=np.arange(0.5, 6.5, 1), alpha=0.75, color='steelblue', edgecolor='black', linewidth=1)
        ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_score:.2f}')

        if original_score_val is not None:
            ax.axvline(original_score_val, color='purple', linestyle='solid', linewidth=2.5, label=f'Original: {original_score_val}')
            max_hist_height = ax.get_ylim()[1]
            # ax.text(original_score_val, max_hist_height * 0.9, "Orig", color='purple', fontsize=8, fontweight='bold', ha='center', va='top',
            #        bbox=dict(facecolor='white', alpha=0.5, pad=0.2))

        title_str = f'Sample {str(sample_id)}'
        if original_score_val is not None: title_str += f' (Orig: {original_score_val})'
        ax.set_title(title_str, fontsize=9)
        ax.set_xlabel('Rating', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3, linestyle=':')
        if i == 0: ax.legend(fontsize=7, loc='upper right')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f'Rating Distributions for {n_samples} Samples ( Adversarial Runs )', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust for suptitle and bottom
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Histogram grid saved to {output_path}")


def analyze_adversarial_vs_normal_scores(logger, results: List[Dict[str, Any]], original_scores: Dict[str, int], output_dir:str = "", output_basename:str="adversarial_vs_normal_analysis"):
    output_path = os.path.join(output_dir, f"{output_basename}.json")
    viz_path = os.path.join(output_dir, f"{output_basename}.png")

    if not results or not original_scores:
        logger.warning("No results or original scores available for comparison.")
        return

    score_pairs = []
    for result in results:
        sample_id = result.get('id')
        if not sample_id or sample_id not in original_scores:
            continue
        original_score = original_scores[sample_id]
        adversarial_score = result.get('adversarial_score')
        if original_score is not None and adversarial_score is not None and adversarial_score != float('inf'):
            score_pairs.append((original_score, adversarial_score))

    if not score_pairs:
        logger.warning("No valid score pairs found for comparison.")
        return

    original_scores_list, adversarial_scores_list = zip(*score_pairs)
    avg_original = np.mean(original_scores_list)
    avg_adversarial = np.mean(adversarial_scores_list)
    median_original = np.median(original_scores_list)
    median_adversarial = np.median(adversarial_scores_list)
    differences = [orig - adv for orig, adv in score_pairs]
    avg_difference = np.mean(differences)
    worse_count = sum(1 for orig, adv in score_pairs if adv < orig)
    same_count = sum(1 for orig, adv in score_pairs if adv == orig)
    better_count = sum(1 for orig, adv in score_pairs if adv > orig)

    analysis_data = { # Renamed
        "sample_count": len(score_pairs),
        "original_scores_stats": { # Renamed for clarity
            "average": float(avg_original), "median": float(median_original),
            "min": float(min(original_scores_list)), "max": float(max(original_scores_list))
        },
        "adversarial_scores_stats": { # Renamed
            "average": float(avg_adversarial), "median": float(median_adversarial),
            "min": float(min(adversarial_scores_list)), "max": float(max(adversarial_scores_list))
        },
        "comparison_stats": { # Renamed
            "average_difference_orig_minus_adv": float(avg_difference), # Clarified
            "count_adv_worse_than_orig": worse_count, "percent_adv_worse": float(worse_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
            "count_adv_same_as_orig": same_count, "percent_adv_same": float(same_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
            "count_adv_better_than_orig": better_count, "percent_adv_better": float(better_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)

    plt.figure(figsize=(14, 12)) # Adjusted figure size

    plt.subplot(2, 2, 1)
    bp_data = [original_scores_list, adversarial_scores_list] # Renamed for clarity
    plt.boxplot(bp_data, labels=['Original Scores', 'Adversarial Scores'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2))
    plt.title('Score Distribution: Original vs. Adversarial', fontsize=12)
    plt.ylabel('Score (1-5)', fontsize=10)
    plt.yticks(np.arange(min(min(original_scores_list, default=1), min(adversarial_scores_list, default=1)), max(max(original_scores_list, default=5), max(adversarial_scores_list, default=5)) + 1))
    plt.grid(True, linestyle=':', alpha=0.7)
    # Add means
    plt.scatter([1, 2], [avg_original, avg_adversarial], color='darkred', marker='o', s=50, zorder=3, label=f'Means:\nOrig: {avg_original:.2f}\nAdv: {avg_adversarial:.2f}')
    plt.legend(fontsize=8)

    plt.subplot(2, 2, 2)
    plt.hist(differences, bins=np.arange(min(differences)-0.5, max(differences)+1.5, 1) if differences else 10 , alpha=0.75, color='salmon', edgecolor='black')
    plt.axvline(avg_difference, color='darkred', linestyle='--', linewidth=1.5, label=f'Avg Diff: {avg_difference:.2f}')
    plt.title('Histogram of Score Differences (Original - Adversarial)', fontsize=12)
    plt.xlabel('Score Difference (Original - Adversarial)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(2, 2, 3)
    plt.scatter(original_scores_list, adversarial_scores_list, alpha=0.6, color='green', edgecolors='darkgreen')
    min_val_scatter = min(min(original_scores_list, default=1), min(adversarial_scores_list, default=1)) # default values
    max_val_scatter = max(max(original_scores_list, default=5), max(adversarial_scores_list, default=5))
    plt.plot([min_val_scatter, max_val_scatter], [min_val_scatter, max_val_scatter], 'k--', alpha=0.5, label='y=x (No Change)')
    plt.title('Original Score vs. Adversarial Score', fontsize=12)
    plt.xlabel('Original Score', fontsize=10)
    plt.ylabel('Adversarial Score', fontsize=10)
    plt.xticks(np.arange(min_val_scatter, max_val_scatter + 1))
    plt.yticks(np.arange(min_val_scatter, max_val_scatter + 1))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=8)
    plt.axis('equal') # Ensure a square plot for y=x line

    plt.subplot(2, 2, 4)
    bar_labels = ['Orig Avg', 'Adv Avg', 'Orig Med', 'Adv Med']
    bar_values = [avg_original, avg_adversarial, median_original, median_adversarial]
    colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen', 'plum']
    bars = plt.bar(bar_labels, bar_values, color=colors, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    plt.title('Summary Score Statistics', fontsize=12)
    plt.ylabel('Score (1-5)', fontsize=10)
    plt.ylim(0, max(bar_values, default=5) * 1.15) # Dynamic Y limit
    plt.grid(True, linestyle=':', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)
    plt.close()

    logger.info(f"\nAdversarial vs Normal Analysis Results:")
    logger.info(f"  Original average score: {avg_original:.2f}, median: {median_original:.2f}")
    logger.info(f"  Adversarial average score: {avg_adversarial:.2f}, median: {median_adversarial:.2f}")
    logger.info(f"  Average difference (Original - Adversarial): {avg_difference:.2f}")
    logger.info(f"  Samples made worse by adversary: {worse_count}/{len(score_pairs)} ({analysis_data['comparison_stats']['percent_adv_worse']:.1f}%)")
    logger.info(f"  Analysis JSON saved to {output_path}")
    logger.info(f"  Analysis visualization saved to {viz_path}")



