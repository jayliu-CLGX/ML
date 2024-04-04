"""
Visualization Module
--------------------

This module provides a collection of functions designed for plotting various statistical charts and graphs, primarily intended for analyzing the performance of classification models. Each function in this module is tailored for visualizing different aspects of model performance, facilitating a comprehensive evaluation of classification models.

List of Functions:
------------------
1. plot_ks_stats:
   Plots the Kolmogorov-Smirnov statistic plot and optionally saves it. It aids in model validation and statistical analysis by comparing the distribution of scores between the event and non-event groups.
   
2. plot_lift_chart:
   Plots and optionally saves the lift chart, showcasing the model's ability to correctly identify positive instances. It represents the improvement obtained with the model over random guessing.
   
3. plot_gain_chart:
   Plots and optionally saves the gain chart, illustrating the cumulative percentage of positive instances captured by the model as we move down the list of instances, sorted by the predicted probability.
   
4. plot_roc_curve:
   Plots the Receiver Operating Characteristic Curve, calculates, and prints the Gini coefficient. It evaluates the trade-off between the True Positive Rate and the False Positive Rate at various threshold levels.
   
5. plot_precision_recall_curve:
   Plots the Precision-Recall Curve and prints the Area Under the Curve (AUC). It’s particularly useful when the classes are imbalanced, focusing on the performance of the positive class.
   
6. plot_confusion_matrix:
   Visualizes the confusion matrix, showing the number of True Positives, False Positives, True Negatives, and False Negatives. It optionally saves the plot to a file.
   
7. plot_classification_metrics:
   Calculates and plots F1 Score, Precision, and Recall, providing a quick overview of the model's performance across these three metrics. It optionally saves the plot to a file.
   
8. plot_threshold_analysis:
   Plots how precision, recall, and F1 Score vary with different thresholds, providing insight into the optimal threshold for decision-making.
   
9. plot_calibration_curve:
   Plots the Calibration Curve to visualize the reliability of the predicted probabilities, indicating whether the predicted probabilities are well-calibrated with the true probabilities.

Each function in this module comes with its set of parameters allowing for customization of plots, such as defining the plot name and deciding whether to save the plot as a file.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import calibration_curve


def plot_ks_stats(df, multiplier=1, plot_name='ks_plot', save='False'):
    """
    Plots and saves the KS (Kolmogorov-Smirnov) statistic plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the decile, cumpsum, and ks columns.
    mult : float
        Multiplier to scale the 'decile' column for plotting.
    plot_name : str, optional
        Name of the plot which is used to create filename. Defaults to 'ks_plot'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.     
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the KS Plot.
    """
    
    fig, ax = plt.subplots()
    
    event_label = f'Event - Max KS: {df["ks"].max():.4f}'
    ax.plot(df['decile'] * multiplier, df['cumpsum'] * 100, label=event_label)
    ax.plot(df['decile'] * multiplier, df['cumpsum_ne'] * 100, label='Non-Event')
    
    ax.set(xlabel='% of Population', ylabel='% of Responses', title=plot_name)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    ax.legend(loc='best')
    plt.grid()

    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()    
    
    return ax


def plot_lift_chart(df, multiplier=1, plot_name='lift_plot', save='False'):
    """
    Plots and saves the Lift chart
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'decile' and 'cumlift' columns.
    multiplier : float, optional
        Multiplier to scale the 'decile' column for plotting. Defaults to 1.
    plot_name : str, optional
        Name of the plot which is used to create the filename. Defaults to 'lift_plot'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.     
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the Lift Plot.
    """
    
    fig, ax = plt.subplots()
    
    ax.plot(df['decile'] * multiplier, df['cumlift'], label='Lift Curve')
    ax.plot(df['decile'] * multiplier, df['one'], label='Baseline', linestyle='--')
    
    ax.set(xlabel='Percent of Population', ylabel='Lift', title=plot_name)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='best')
    plt.grid()
    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    
    return ax


def plot_roc_curve(df, score, plot_name='roc_plot', label='event', save=False):
    """
    Plots the ROC (Receiver Operating Characteristic) Curve, calculates and prints the Gini 
    coefficient, and optionally saves the plot to a file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'event' column and the score column.
    score : str
        Name of the column in 'df' containing the score values.
    plot_name : str, optional
        Name of the plot which is used as the title and to create the filename. Defaults to 'roc_plot'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the ROC Plot.
    """
    
    y_true = df[label]
    y_pred = df[score]
    # get values
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
           ylabel='True Positive Rate', title=plot_name)
    ax.legend(loc='best')
    plt.grid()
    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    
    gini = (roc_auc - 0.5) / 0.5
    print(f'Gini stat: {gini:.4f}')
    
    return ax

def plot_precision_recall_curve(df, score, plot_name='Precision-Recall Curve', label='event', save=False):
    """
    Plots the Precision-Recall Curve and prints the Area Under the Curve (AUC).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the actual labels ('event') and predicted scores ('score').
    score : str
        The name of the column in 'df' representing the score.
    plot_name : str, optional (default='Precision-Recall Curve')
        The title of the plot.
    save : bool, optional (default=False)
        If True, saves the plot to a file. The filename is created using the 'plot_name'.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the Precision-Recall Curve.
    """
    
    
    y_true = df[label]
    y_pred = df[score]
    # Calculate precision, recall, and AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precision', title=plot_name)
    ax.legend(loc='best')
    plt.grid()
    
    # Save plot if save parameter is True
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    print('PR AUC:', round(pr_auc, 4))
    
    return ax


def plot_confusion_matrix(df, score, class_names=None, plot_name='Confusion Matrix', label='event', save=False):
    """
    Plots the Confusion Matrix and optionally saves it to a file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the actual labels ('event') and predicted scores.
    score : str
        The name of the column in 'df' representing the score.
    class_names : list of str, optional (default=None)
        Names of the classes to be displayed on the x and y axes. 
        If None, classes will be inferred from the 'event' column.
    plot_name : str, optional (default='Confusion Matrix')
        The title of the plot.
    save : bool, optional (default=False)
        If True, saves the plot to a file. The filename is created using the 'plot_name'.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the Confusion Matrix.
    """
    y_true = df[label]
    y_pred = df[score]
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = df['event'].unique().tolist()
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set(xlabel='Predicted label', ylabel='True label', title=plot_name)
    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    return ax


def plot_classification_metrics(df, score, plot_name='Classification Metrics', label='event', save=False):
    """
    Calculates and plots F1 Score, Precision, and Recall, and optionally saves the plot to a file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the actual labels ('event') and predicted scores.
    score : str
        The name of the column in 'df' representing the score.
    plot_name : str, optional
        Name of the plot, used as the title and to create the filename. Defaults to 'Classification Metrics'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.
        
    Returns
    -------
    dict
        A dictionary containing F1 Score, Precision, and Recall.
    """
    
    y_true = df[label]
    y_pred = df[score]
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    metrics_dict = {
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    }
    
    # Print metrics
    print('F1 Score: {:.4f}'.format(f1))
    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    
    # Plot metrics
    fig, ax = plt.subplots()
    
    ax.bar(metrics_dict.keys(), metrics_dict.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set(ylim=[0, 1], title=plot_name)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.show()
    
    # Save the plot if save is True
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
        
    return metrics_dict


def plot_threshold_analysis(y_true, y_scores, plot_name='Threshold Analysis', save=False):
    """
    Plots how precision, recall, and F1 Score vary with different thresholds.
    
    Parameters
    ----------
    y_true : array-like
        True labels of the classification task.
    y_scores : array-like
        Predicted scores by the model.
    plot_name : str, optional
        Name of the plot which is used as the title and to create the filename. Defaults to 'Threshold Analysis'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the Threshold Analysis Plot.
    """

    # Calculate precision, recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 Score for different thresholds
    f1_scores = 2*(precision*recall)/(precision + recall)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precision[:-1], label='Precision')
    ax.plot(thresholds, recall[:-1], label='Recall')
    ax.plot(thresholds, f1_scores[:-1], label='F1 Score')
    
    ax.set(xlabel='Threshold', ylabel='Score', title=plot_name)
    ax.legend(loc='best')
    plt.grid()
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
        
    return ax


def plot_calibration_curve(y_true, y_probs, n_bins=10, plot_name='Calibration Curve', save=False):
    """
    Plots the Calibration Curve to visualize the reliability of the predicted probabilities.
    
    Parameters
    ----------
    y_true : array-like
        True labels of the classification task.
    y_probs : array-like
        Predicted probabilities by the model.
    n_bins : int, optional
        Number of bins to use for the calibration_curve. Defaults to 10.
    plot_name : str, optional
        Name of the plot which is used as the title and to create the filename. Defaults to 'Calibration Curve'.
    save : bool, optional
        Whether to save the plot to a file. Defaults to False.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Axes object with the Calibration Curve.
    """
    
    
    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='o', color='b', label='Model')
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives', title=plot_name)
    ax.legend(loc='best')
    
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
        
    return ax


# def compare_model_gains_and_fpr(df1, df2, target, segment_list=[1, 5, 10, 15, 20], 
#                                 model_names=['Retrained', 'Current'], plot_name='model_comparison', save=False):
#     """
#     Compares the gain and false positive ratios of two different models and plots the results.
    
#     This function takes two DataFrames representing the scores from two different models, calculates the gain values 
#     and false positive ratios for the specified top populations, and plots the results for comparison. 
#     It returns a DataFrame containing the calculated gain values and false positive ratios.
    
#     Parameters
#     ----------
#     df1 : pd.DataFrame
#         DataFrame containing the scores from the first model.
#     df2 : pd.DataFrame
#         DataFrame containing the scores from the second model.
#     target : str
#         The name of the target variable column in df1 and df2.
#     segment_list : list of int, optional
#         List of top populations in percentage to be considered for the gain chart.
#         Defaults to [1, 5, 10, 15, 20].
#     plot_name : str, optional
#         Name of the plot which is used as the title and to create the filename. Defaults to 'model_comparison'.
#     save : bool, optional
#         Whether to save the plot to a file. Defaults to False.
        
#     Returns
#     -------
#     results_df : pd.DataFrame
#         A DataFrame containing the calculated gain values and false positive ratios.
#     """
    
#     results_df = pd.DataFrame(segment_list, columns=['Top Population %'])
    
#     for df, label in zip([df1, df2], model_names):
#         df_sorted = df.sort_values(by=target, ascending=False).reset_index(drop=True)
#         gains = []
#         fprs = []  # False Positive Ratios
#         for segment in segment_list:
#             top_population = int(len(df) * (segment / 100))
#             segment_df = df_sorted.loc[:top_population]
            
#             # Calculating gains
#             captured_target = segment_df[target].sum() / df_sorted[target].sum()
#             gains.append(captured_target * 100)
            
#             # Calculating false positive ratios
#             FP = len(segment_df) - segment_df[target].sum()
#             TN = len(df_sorted) - top_population - (df_sorted[target].sum() - segment_df[target].sum())
#             fpr = FP / (TN + FP)
#             fprs.append(fpr * 100)
        
#         results_df[f'Target Captured {label} %'] = gains
#         results_df[f'False Positive Ratio {label} %'] = fprs
    
#     # Plotting
#     fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
#     results_df.plot(x='Top Population %', y=[col for col in results_df.columns if 'Target Captured' in col], 
#                     linestyle='-', marker='o', ax=axs[0])
#     axs[0].set_ylabel('Target Captured %')
#     axs[0].grid(True)
    
#     results_df.plot(x='Top Population %', y=[col for col in results_df.columns if 'False Positive Ratio' in col], 
#                     linestyle='-', marker='o', ax=axs[1])
#     axs[1].set_ylabel('False Positive Ratio %')
#     axs[1].grid(True)
    
#     plt.tight_layout()
    
#     if save:
#         filename = f"{plot_name.replace(' ', '_')}.png"
#         fig.savefig(filename, bbox_inches='tight')
    
#     plt.show()
    
#     return axes, results_df

# def plot_gain_chart_and_fpr(df, target, multiplier=1, plot_name='gain_plot', save=False):
#     """
#     Plots the Gain and False Positive Ratio Curves and optionally saves it to a file.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing the score and target columns.
#     target : str
#         The name of the target variable column in df.
#     multiplier : float, optional
#         Multiplier to scale the 'decile' column for plotting. Defaults to 1.
#     plot_name : str, optional
#         Name of the plot which is used as the title and to create the filename. Defaults to 'gain_plot'.
#     save : bool, optional
#         Whether to save the plot to a file. Defaults to False.
        
#     Returns
#     -------
#     matplotlib.axes._subplots.AxesSubplot
#         Axes object with the Gain and FPR Plot.
#     results_df : pd.DataFrame
#         DataFrame containing the gain and false positive ratio values.
#     """
    
#     df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)
#     total_population = len(df_sorted)
#     total_targets = df_sorted[target].sum()
    
#     results_df = pd.DataFrame()
#     gains = []
#     fprs = []
    
#     for decile in df['decile'].unique():
#         top_population = int(total_population * (decile / 100))
#         segment_df = df_sorted.loc[:top_population]
        
#         # Calculating gains
#         captured_target = segment_df[target].sum() / total_targets
#         gains.append(captured_target * 100)
        
#         # Calculating false positive ratios
#         FP = len(segment_df) - segment_df[target].sum()
#         TN = total_population - top_population - (total_targets - segment_df[target].sum())
#         fpr = FP / (TN + FP) if (TN + FP) != 0 else 0
#         fprs.append(fpr * 100)
    
#     results_df['Top Population %'] = df['decile'].unique() * multiplier
#     results_df['Target Captured %'] = gains
#     results_df['False Positive Ratio %'] = fprs
    
#     # Plotting
#     fig, ax1 = plt.subplots(figsize=(10, 6))
    
#     color = 'tab:blue'
#     ax1.set_xlabel('Top Population %')
#     ax1.set_ylabel('Target Captured %', color=color)
#     ax1.plot(results_df['Top Population %'], results_df['Target Captured %'], color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
    
#     ax2 = ax1.twinx()  
#     color = 'tab:red'
#     ax2.set_ylabel('False Positive Ratio %', color=color)  
#     ax2.plot(results_df['Top Population %'], results_df['False Positive Ratio %'], color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
    
#     plt.title(plot_name)
#     plt.grid()
    
#     if save:
#         filename = f"{plot_name.replace(' ', '_')}.png"
#         fig.savefig(filename, bbox_inches='tight')
    
#     plt.show()
    
#     return ax1, results_df


# def calculate_gains_and_fprs(df, target, segment_list=[1, 5, 10, 15, 20]):
#     """
#     Calculate gain and false positive ratios for given segments of a DataFrame.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing the scores.
#     target : str
#         The name of the target variable column in df.
#     segment_list : list of int
#         List of top populations in percentage to be considered for the gain and FPR calculations.

#     Returns
#     -------
#     results_df : pd.DataFrame
#         A DataFrame containing the calculated gain values and false positive ratios.
#     """
#     df_sorted = df.sort_values(by=target, ascending=False).reset_index(drop=True)
#     total_population = len(df_sorted)
#     total_targets = df_sorted[target].sum()
    
#     results_df = pd.DataFrame()
#     gains = []
#     fprs = []

#     for segment in segment_list:
#         top_population = int(total_population * (segment / 100))
#         segment_df = df_sorted.loc[:top_population]
        
#         # Calculating gains
#         captured_target = segment_df[target].sum() / total_targets
#         gains.append(captured_target * 100)
        
#         # Calculating false positive ratios
        
#         # TP = segment_df[target].sum()
#         FP = len(segment_df) - segment_df[target].sum()
#         # print(segment_df[target].shape, TP, FP)
#         TN = total_population - top_population - (total_targets - segment_df[target].sum())
#         # TP = (segment_df[target] == 0).sum()
#         fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
#         fprs.append(fpr * 100)
    
#     results_df['Top Population %'] = segment_list
#     results_df['Target Captured %'] = gains
#     results_df['False Discovery Rate %'] = fprs
    
#     return results_df


import pandas as pd

def calculate_gains_and_fprs(df, target, score_col, segment_list=[1, 5, 10, 15, 20]):
    """
    Calculate gain and false positive ratios for given segments of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the scores.
    target : str
        The name of the target variable column in df.
    score_col : str
        The name of the score/probability column used to sort and segment the data.
    segment_list : list of int
        List of top populations in percentage to be considered for the gain and FPR calculations.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame containing the calculated gain values and false positive ratios.
    """
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    total_population = len(df_sorted)
    total_targets = df_sorted[target].sum()
    TP_tot = sum(df_sorted[target] == 1)
    FP_tot = sum(df_sorted[target] == 0)
    FP_over_TP_tot = round((FP_tot/TP_tot), 2)
    results_df = pd.DataFrame()
    gains = []
    fprs = []

    for segment in segment_list:
        top_population = int(total_population * (segment / 100))
        segment_df = df_sorted.iloc[:top_population]
        
        # Calculating gains
        captured_target = segment_df[target].sum() / total_targets
        gains.append((captured_target * 100).round(1))
        
        # Calculating false positive ratios
        TP = segment_df[target].sum()
        FP = len(segment_df) - TP
        TN = total_population - top_population - (total_targets - TP)
        
        
        fpr = FP / (TP) if (TP) != 0 else 0
        fprs.append(f'{fpr.round(2)} ({FP_over_TP_tot})')
    
    results_df['Top Population %'] = segment_list
    results_df['Target Captured %'] = gains
    results_df['FP / TP'] = fprs
    
    return results_df


def compare_model_gains_and_fpr(df1, df2, target, score_col, segment_list=[1, 5, 10, 15, 20], 
                                model_names=['Retrained', 'Current'], plot_name='model_comparison', save=False):
    """
    [Your Existing Docstring]
    """
    results_df = pd.DataFrame(segment_list, columns=['Top Population %'])
    
    for df, label in zip([df1, df2], model_names):
        temp_results_df = calculate_gains_and_fprs(df, target, score_col, segment_list)
        results_df[f'Target Captured {label} %'] = temp_results_df['Target Captured %']
        results_df[f'False Positive Ratio {label} %'] = temp_results_df['False Positive Ratio %']
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    results_df.plot(x='Top Population %', y=[col for col in results_df.columns if 'Target Captured' in col], 
                    linestyle='-', marker='o', ax=axs[0])
    axs[0].set_ylabel('Target Captured %')
    axs[0].grid(True)
    
    results_df.plot(x='Top Population %', y=[col for col in results_df.columns if 'False Positive Ratio' in col], 
                    linestyle='-', marker='o', ax=axs[1])
    axs[1].set_ylabel('False Positive Ratio %')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    
    return results_df, axs  # axs is your array of Axes object containing the subplots



def print_gain_table(dec_df, multiplier=1):
    """
    Generate a gain table.

    Parameters:
    - df: DataFrame containing the dec dataframe
      it should be created by propensity_data_process.create_dec_table(data, score, split)
    - mult: A multiplier for the decile

    Returns:
    A DataFrame representing the gain table.
    """
    # Create a new DataFrame for the gain table
    gain_df = pd.DataFrame()
    # % of Population
    gain_df['Top Scoring %'] = df['decile'] * multiplier
    gain_df['min_score'] = df['min_score']
    gain_df['Count of Positive Outcomes'] = df['cumsum']
    gain_df['Count of Negative Outcomes'] = df['cumcount'] - df['cumsum']
    gain_df['Total Solicited'] = df['cumcount']
    gain_df['% of Total Positive Outcomes'] = round(df['cumpsum'] * 100, 2)
    gain_df['Odds'] = round ((df['cumcount'] - df['cumsum']) / df['cumsum'], 2)
    gain_df['List Rate %'] = round(df['sum'] / df['count'] * 100, 2)
    gain_df.reset_index(inplace=True, drop=True)
    return gain_df


import matplotlib.pyplot as plt

def plot_gain_chart_and_fpr(df, target, score_col, segments=[1, 5, 10, 15, 20], plot_name='gain_plot', save=False):
    """
    Plot Gain and FPR chart for specified segments.
    
    Parameters:
    - df: DataFrame containing the data
    - target: String, name of the target column
    - segments: List of integers, specifying the top population percentages to calculate gains and FPR for
    - plot_name: String, title of the plot
    - save: Boolean, whether to save the plot as a .png file
    
    Returns:
    - results_df: DataFrame containing the calculated gains and FPRs
    - ax1: Axes object containing the plot
    """
    # Calculate gains and FPRs
    results_df = calculate_gains_and_fprs(df, target, score_col, segments)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Top Population %')
    ax1.set_ylabel('Target Captured %', color=color)
    ax1.plot(results_df['Top Population %'], results_df['Target Captured %'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('False Positive Ratio %', color=color)  
    ax2.plot(results_df['Top Population %'], results_df['False Positive Ratio %'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(plot_name)
    plt.grid()
    
    if save:
        filename = f"{plot_name.replace(' ', '_')}.png"
        fig.savefig(filename, bbox_inches='tight')
    
    plt.show()
    
    return results_df, ax1  

import pandas as pd

def calculate_gains_and_fprs_for_target_capture(df, target, target_capture_segments=[1, 5, 10, 15, 20]):
    """
    Calculate gain and false positive ratios for given segments based on target capture.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the scores.
    target : str
        The name of the target variable column in df.
    target_capture_segments : list of int
        List of target captured percentages to be considered for the gain and FPR calculations.
    
    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame containing the calculated gain values and false positive ratios.
    """
    df_sorted = df.sort_values(by=target, ascending=False).reset_index(drop=True)
    total_population = len(df_sorted)
    total_targets = df_sorted[target].sum()
    
    results_df = pd.DataFrame()
    gains = []
    fprs = []
    non_event_ratios = []
    
    for capture_percentage in target_capture_segments:
        target_count_to_reach = (capture_percentage / 100) * total_targets
        
        cumulative_targets = df_sorted[target].cumsum()
        
        # Finding the index (top segment) where the desired capture_percentage is reached or exceeded
        reach_idx = cumulative_targets[cumulative_targets >= target_count_to_reach].index.min()
        
        segment_df = df_sorted.loc[:reach_idx]
        
        # Calculating gains
        captured_target = segment_df[target].sum() / total_targets
        gains.append(captured_target * 100)
        
        # Calculating false positive ratios
        FP = len(segment_df) - segment_df[target].sum()
        TN = total_population - len(segment_df) - (total_targets - segment_df[target].sum())
        fpr = FP / (TN + FP) if (TN + FP) != 0 else 0
        fprs.append(fpr * 100)
        
        # Calculating Non-event to event ratio in the top segment
        non_event_per_event = FP / segment_df[target].sum() if segment_df[target].sum() != 0 else 0
        non_event_ratios.append(non_event_per_event)
    
    results_df['Target Captured %'] = target_capture_segments
    results_df['Actual Target Captured %'] = gains
    results_df['False Positive Ratio %'] = fprs
    results_df['Non-event to Event Ratio'] = non_event_ratios
    
    return results_df
