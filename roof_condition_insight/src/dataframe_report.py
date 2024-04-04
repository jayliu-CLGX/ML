import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import logging
import itertools 
import numpy as np
from collections import Counter
from scipy.stats import entropy

def compute_statistics(data, features):
    """
    Compute various statistics for specified features in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the features for which the statistics are to be calculated.
    features : list of str
        List of feature names for which statistics are to be calculated.

    Returns
    -------
    dict
        A dictionary where keys are feature names and values are dictionaries with statistic names as keys and 
        corresponding calculated values as values.
    """
    stats = {}
    for feature in features:
        feature_data = data[feature]
        
        feature_type = 'Numeric' if pd.api.types.is_numeric_dtype(feature_data) else 'Categorical'
        
        stats[feature] = {
            'type': feature_type,
            'count': feature_data.shape[0],  # Count of non-NaN values
            'null_percentage': (100 *feature_data.isna().sum() / feature_data.shape[0]).round(2), # Count of NaN values
            'top_values': feature_data.value_counts().head(5).to_dict(),
            'unique_values': feature_data.nunique()
        }

        
        if pd.api.types.is_numeric_dtype(feature_data):
            stats[feature].update({
                'mean': round(feature_data.mean(), 2),
                'median': round(feature_data.median(), 2),
                '25th percentile': round(feature_data.quantile(0.25), 2),
                '75th percentile': round(feature_data.quantile(0.75), 2),
                'std': round(feature_data.std(), 2),
                'min': round(feature_data.min(), 2),
                'max': round(feature_data.max(), 2),
            })
        else:
            stats[feature].update({
                'mean': 'N/A',
                'median': 'N/A',
                '25th percentile': 'N/A',
                '75th percentile': 'N/A',
                'std': 'N/A',
                'min': 'N/A',
                'max': 'N/A'
            })
    return stats

# def create_stats_df(dataframes, features=None, suffixes=None):
#     """
#     Create a dataframe containing various statistics for the specified features in the given dataframes.
    
#     Parameters
#     ----------
#     dataframes : pd.DataFrame or list of pd.DataFrame
#         DataFrame or list of DataFrames to compute statistics on.
#     features : list of str, optional
#         List of feature names (str) to compute statistics for.
#         If None, all columns in the first dataframe in dataframes are used. 
#         Defaults to None.
#     suffixes : list of str, optional
#         List of suffixes (str) corresponding to each DataFrame in dataframes.
#         If None, no suffixes are added to the keys in the resulting dataframe.
#         Defaults to None.
    
#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing computed statistics with features as rows and statistics as columns.
#     """
    
#     # Ensure dataframes is a list even if a single DataFrame is passed
#     if not isinstance(dataframes, list):
#         dataframes = [dataframes]
#         if suffixes is not None:  # Ensure suffixes is a list of the same length as dataframes
#             suffixes = [suffixes]
    
#     if features is None:
#         features = dataframes[0].columns.tolist()
    
#     if suffixes is None:
#         suffixes = [''] * len(dataframes)
    
#     all_stats = {}
#     for dataframe, suffix in zip(dataframes, suffixes):
#         stats = compute_statistics(dataframe, features)
#         updated_keys = {f"{key}{suffix}": value for key, value in stats.items()}
#         all_stats.update(updated_keys)
    
#     # Compute statistics for the difference of features if more than one dataframe is provided
#     if len(dataframes) > 1:
#         diff_feature_stats = compute_statistics(dataframes[0], [f"{feature}_diff" for feature in features])
#         all_stats.update(diff_feature_stats)
    
#     stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
#     return stats_df.sort_index()

def create_stats_df(dataframes, features=None, suffixes=None):
    """
    Create a dataframe containing various statistics for the specified features in the given dataframes.
    
    Parameters
    ----------
    dataframes : pd.DataFrame or list of pd.DataFrame
        DataFrame or list of DataFrames to compute statistics on.
    features : list of str, optional
        List of feature names (str) to compute statistics for.
        If None, all columns in the first dataframe in dataframes are used. 
        Defaults to None.
    suffixes : list of str, optional
        List of suffixes (str) corresponding to each DataFrame in dataframes.
        If None, no suffixes are added to the keys in the resulting dataframe.
        Defaults to None.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing computed statistics with features as rows and statistics as columns.
    """

    def contains_non_hashable(column):
        for item in column:
            if isinstance(item, (list, dict, set)):
                return True
        return False

    def preprocess_dataframe(dataframe):
        # Convert list, dict, and set columns to string
        for col in dataframe.columns:
            if contains_non_hashable(dataframe[col]):
                print(f"Column '{col}' contains non-hashable types. Converting to string.")
                dataframe[col] = dataframe[col].apply(lambda x: str(x) if isinstance(x, (list, dict, set)) else x)
                
        # Convert boolean columns to integer
        for col in dataframe.columns:
            if dataframe[col].dtype == 'bool':
                print(f"Column '{col}' converted from bool to int.")
                dataframe[col] = dataframe[col].astype(int)
        return dataframe
    
    # Ensure dataframes is a list even if a single DataFrame is passed
    if not isinstance(dataframes, list):
        dataframes = [dataframes]
    if suffixes is not None and not isinstance(suffixes, list):  # Ensure suffixes is a list of the same length as dataframes
        suffixes = [suffixes]
    if features is None:
        features = dataframes[0].columns.tolist()
    if suffixes is None:
        suffixes = [''] * len(dataframes)
    
    all_stats = {}
    for dataframe, suffix in zip(dataframes, suffixes):
        dataframe = preprocess_dataframe(dataframe.copy())  # Preprocess and make conversions
        stats = compute_statistics(dataframe, features)
        updated_keys = {f"{key}{suffix}": value for key, value in stats.items()}
        all_stats.update(updated_keys)
    
    # Compute statistics for the difference of features if more than one dataframe is provided
    if len(dataframes) > 1:
        diff_feature_stats = compute_statistics(dataframes[0], [f"{feature}_diff" for feature in features])
        all_stats.update(diff_feature_stats)
    
    stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
    
    # Sorting to have type numeric on top of the dataframe
    stats_df['sort_key'] = stats_df['type'].apply(lambda x: 0 if x == 'Numeric' else 1)
    stats_df = stats_df.sort_values(by=['sort_key']).drop('sort_key', axis=1)
    
    return stats_df.sort_index()


def calculate_psi(expected, actual, bins=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    """
    expected_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)

    # Avoid division by zero
    expected_hist += 0.0001
    actual_hist += 0.0001

    psi = np.sum((expected_hist - actual_hist) * np.log(expected_hist / actual_hist))
    return psi

def calculate_ksi(expected, actual, bins=10):
    """
    Calculate the Kullback-Leibler Stability Index (KSI) between two distributions.
    """
    expected_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)

    # Avoid division by zero
    expected_hist += 0.0001
    actual_hist += 0.0001

    ksi = np.sum(actual_hist * np.log(actual_hist / expected_hist))
    return ksi

def calculate_categorical_psi(expected, actual):
    # Get the union of all categories in expected and actual
    all_categories = set(expected).union(set(actual))
    
    # Generate the count dictionary for expected and actual
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)

    # Generate probability arrays for each category in all_categories
    expected_probs = np.array([expected_counts[cat] / len(expected) for cat in all_categories])
    actual_probs = np.array([actual_counts[cat] / len(actual) for cat in all_categories])

    # Avoid division by zero
    expected_probs += 0.0001
    actual_probs += 0.0001

    # Calculating PSI
    psi = np.sum((expected_probs - actual_probs) * np.log(expected_probs / actual_probs))
    
    return psi

def calculate_categorical_ksi(expected, actual):
    # Get the union of all categories in expected and actual
    all_categories = set(expected).union(set(actual))
    
    # Generate the count dictionary for expected and actual
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)

    # Generate probability arrays for each category in all_categories
    expected_probs = np.array([expected_counts.get(cat, 0) / len(expected) for cat in all_categories])
    actual_probs = np.array([actual_counts.get(cat, 0) / len(actual) for cat in all_categories])

    # Avoid division by zero
    expected_probs += 0.0001
    actual_probs += 0.0001

    # Calculating KSI
    ksi = np.sum(actual_probs * np.log(actual_probs / expected_probs))
    
    return ksi



def drift_analysis(df1, df2):
    """
    Compute drift between two dataframes for each feature.
    """
    drift_data = []

    for column in df1.columns:
        # Common calculations
        count1, count2 = df1[column].count(), df2[column].count()
        null_count1, null_count2 = df1[column].isnull().sum(), df2[column].isnull().sum()
        unique1, unique2 = df1[column].nunique(), df2[column].nunique()
        top_values1 = df1[column].value_counts().head(5).to_dict()
        top_values2 = df2[column].value_counts().head(5).to_dict()

        if pd.api.types.is_numeric_dtype(df1[column]):
            # Calculate drift for numeric columns
            mean1, mean2 = df1[column].mean(), df2[column].mean().round(2)
            median1, median2 = df1[column].median(), df2[column].median().round(2)
            std1, std2 = df1[column].std().round(2), df2[column].std().round(2)
            psi = calculate_psi(df1[column], df2[column]).round(4)
            ksi = calculate_ksi(df1[column], df2[column]).round(4)


            drift_data.append({
                'Feature': column,
                'Type': 'Numeric',
                'Count_df1': count1,
                'Count_df2': count2,
                'Null_Count_df1': null_count1,
                'Null_Count_df2': null_count2,
                'Unique_Values_df1': unique1,
                'Unique_Values_df2': unique2,
                'Top5_Values_df1': top_values1,
                'Top5_Values_df2': top_values2,
                'Mean_df1': mean1,
                'Mean_df2': mean2,
                'Median_df1': median1,
                'Median_df2': median2,
                'StdDev_df1': std1,
                'StdDev_df2': std2,
                'PSI': psi,
                'KSI': ksi
            })
        else:
            # Calculate drift for categorical columns
            freq1 = df1[column].value_counts(normalize=True).to_dict()
            freq2 = df2[column].value_counts(normalize=True).to_dict()
            psi = calculate_categorical_psi(df1[column], df2[column]).round(4)
            ksi = calculate_categorical_ksi(df1[column], df2[column]).round(4)
            
            # Fill missing keys in freq1 and freq2
            for key in freq1.keys():
                freq2[key] = freq2.get(key, 0)
            for key in freq2.keys():
                freq1[key] = freq1.get(key, 0)

            # Calculate the difference in proportions for each category
            diff = {k: abs(freq1[k] - freq2[k]) for k in freq1.keys()}

            drift_data.append({
                'Feature': column,
                'Type': 'Categorical',
                'Count_df1': count1,
                'Count_df2': count2,
                'Null_Count_df1': null_count1,
                'Null_Count_df2': null_count2,
                'Unique_Values_df1': unique1,
                'Unique_Values_df2': unique2,
                'Top5_Values_df1': top_values1,
                'Top5_Values_df2': top_values2,
                'Diff_Proportions': diff,
                'PSI': psi,
                'KSI': ksi
            })

    drift_df = pd.DataFrame(drift_data)
    return drift_df

def compare_score_distributions(dfs, score_column, labels, colors=None):
    """
    Compare the score distributions among multiple dataframes by plotting descriptive 
    statistics and box plots for the specified column.
    
    Parameters
    ----------
    dfs : list of DataFrame
        The list of dataframes for comparison.
    score_column : str
        The name of the score column to compare in all dataframes.
    labels : list of str
        A list containing the labels for each DataFrame's plot.
    colors : list of str, optional
        A list containing the colors for each DataFrame's plot. If None, default colors are used.
        
    Returns
    -------
    None
        The function only displays the plot and doesn't return any value.
        
    Examples
    --------
    >>> compare_score_distributions_multiple_dfs(
    ...     dfs=[refi_national_resampled_df, refi_flagstar_df], 
    ...     score_column='Propensity_Score_Refinance_Model', 
    ...     labels=['National', 'Flagstar']
    ... )
    """
    
    if len(dfs) != len(labels):
        raise ValueError("The length of dfs and labels must be equal.")
    
    # Define some default colors
    default_colors = ['lightblue', '#FFA500', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow', 'lightgray']
    color_cycle = itertools.cycle(default_colors)
    
    if colors is None:
        colors = [next(color_cycle) for _ in range(len(dfs))]
    
    df_combined = pd.DataFrame()
    stats_df = pd.DataFrame()
    
    for df, label, color in zip(dfs, labels, colors):
        if score_column not in df.columns:
            raise ValueError(f"The specified score_column '{score_column}' does not exist in DataFrame with label '{label}'.")
        
        # Calculating descriptive statistics for each DataFrame
        desc_stats_df = df[score_column].describe()
        stats_df = pd.concat([stats_df, desc_stats_df.rename(label)], axis=1)
        
        df_combined = pd.concat([df_combined, df[score_column].rename(label)], axis=1)

    # Creating a subplot grid of 1 row x 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Adjust figsize to your needs

    # Plotting descriptive statistics in the first column
    ax[0].axis('tight')
    ax[0].axis('off')
    table_data = []
    columns = ['Statistic'] + labels
    for stat in stats_df.index:
        row_data = [stat] + ['{:.2f}'.format(x) if stat in ['mean', 'std'] else x for x in stats_df.loc[stat].tolist()]
        table_data.append(row_data)
    ax[0].table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
    ax[0].set_title('Descriptive Statistics')

    # Boxplot for the new combined DataFrame in the second column
    df_combined.boxplot(ax=ax[1], patch_artist=True, boxprops=dict(facecolor='lightgrey'))
    ax[1].set_title(f"Box plot of '{score_column}'")
    ax[1].set_ylabel("Score Value")
    plt.xticks(rotation=10)

    plt.tight_layout()
    plt.show()


def plot_histograms(dataframes, column_name, labels, colors=None, xrange=None , bins='auto'):
    if len(dataframes) != len(labels):
        raise ValueError("The length of dataframes and labels must be equal.")
    
    # Define some default colors if not provided
    if colors is None:
        default_colors = ['lightblue', '#FFA500', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow', 'lightgray']
        color_cycle = itertools.cycle(default_colors)  # create a cycle of colors
        colors = [next(color_cycle) for _ in range(len(dataframes))]  # assign colors from the color cycle
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    
    for df, label, color in zip(dataframes, labels, colors):
        if column_name not in df.columns:
            raise ValueError(f"The specified column_name '{column_name}' does not exist in DataFrame with label '{label}'.")

        # Construct histplot arguments dynamically
        histplot_args = {
            'element': "step",
            'stat': "density",
            'common_norm': False,
            'label': label,
            'color': color,
            'kde': True,
            'bins': bins
        }
        
        # Only add binrange if both xmin and xmax are specified
        if xrange is not None:
            histplot_args['binrange'] = xrange
        
        sns.histplot(df[column_name], **histplot_args)
    
    plt.title(f'Histogram of "{column_name}" with KDE', fontsize=15)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

def compare_feature_histograms_tags(dfs, df_names, features, target, axis_type='percentage'):
    """
    Compare histograms of specified features for a list of dataframes, with option for percentage or frequency on the y-axis.
    
    Parameters:
    - dfs (list of pd.DataFrame): List of DataFrames containing the data to be plotted
    - df_names (list of str): Names of the DataFrames for labeling
    - features (list of str): List of feature strings to be plotted
    - target (str): String indicating the target variable
    - axis_type (str): Type of y-axis ('percentage' or 'frequency')
    """
    assert len(dfs) == len(df_names), "The length of dfs and df_names should be equal"
    assert axis_type in ['percentage', 'frequency'], "axis_type must be 'percentage' or 'frequency'"
    
    for feature in features:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        
        def plot_and_annotate(ax, data, bar_colors, text_colors, names, title):
            max_y_value = 0  

            for idx, (d, bar_color, text_color, name) in enumerate(zip(data, bar_colors, text_colors, names)):
                if axis_type == 'percentage':
                    weights = [1./len(d)]*len(d)
                else:
                    weights = None
                
                count, bins, _ = ax.hist(d, bins=30, edgecolor='k', alpha=0.5, label=f'{name} (NaN: {nan_percent[idx]:.2f}%)', weights=weights, color=bar_color)

                if axis_type == 'percentage':
                    for c, b in zip(count, bins):
                        if c > 0:  
                            ax.text(b+0.01*(max(bins)-min(bins)), c, f"{int(round(c * len(d)))}", color=text_color, ha='left', va='bottom')
                            if c > max_y_value:
                                max_y_value = c
                elif axis_type == 'frequency':
                    max_y_value = max(count)  # directly use max count since it’s the frequency

                ax.set_title(title)
                ax.set_xlabel('Value')
                ax.set_ylabel('Percentage' if axis_type == 'percentage' else 'Frequency')
                ax.legend(loc='upper right')
                ax.set_ylim(0, max_y_value + max_y_value*0.1)

                if axis_type == 'percentage':
                    ax.yaxis.set_major_formatter(PercentFormatter(1))

        
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Darker colors for bars
        text_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']  # Corresponding lighter colors for text annotations

        nan_percent = [(d[feature].isna().sum() / len(d)) * 100 for d in dfs]
        data_overall = [d[feature].dropna() for d in dfs]
        
        plot_and_annotate(axs[0], data_overall, bar_colors, text_colors, df_names, f'Overall {feature}')
        data_target0 = [d[d[target] == 0][feature].dropna() for d in dfs]
        plot_and_annotate(axs[1], data_target0, bar_colors, text_colors, df_names, f'{target} = 0, {feature}')
        data_target1 = [d[d[target] == 1][feature].dropna() for d in dfs]
        plot_and_annotate(axs[2], data_target1, bar_colors, text_colors, df_names, f'{target} = 1, {feature}')
        
        plt.show()
        
        
def compare_feature_boxplots(dfs, df_names, features, target):
    """
    Generate box plots of specified features for a list of dataframes, compared overall and for two target categories.
    
    Parameters:
    - dfs (list of pd.DataFrame): List of DataFrames containing the data to be plotted.
    - df_names (list of str): Names of the DataFrames for labeling.
    - features (list of str): List of feature strings to be plotted.
    - target (str): String indicating the target variable.
    """
    assert len(dfs) == len(df_names), "The length of dfs and df_names should be equal"
    
    for feature in features:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)  
        
        nan_percents = [(df[feature].isna().sum() / len(df)) * 100 for df in dfs]
        data_overall = [df[feature].dropna() for df in dfs]
        data_target0 = [df[df[target] == 0][feature].dropna() for df in dfs]
        data_target1 = [df[df[target] == 1][feature].dropna() for df in dfs]
        
        def plot_and_annotate(ax, data, names, title):
            ax.boxplot(data, labels=[f'{name}\n(NaN: {nan_percent:.2f}%)' for name, nan_percent in zip(names, nan_percents)])
            ax.set_title(title)
            ax.set_ylabel('Value')

        plot_and_annotate(axs[0], data_overall, df_names, f'Overall {feature}')
        plot_and_annotate(axs[1], data_target0, df_names, f'{target} = 0, {feature}')
        plot_and_annotate(axs[2], data_target1, df_names, f'{target} = 1, {feature}')
        
        plt.show()
        
        
def plot_date_difference_histogram(df, date_col1, date_col2, condition_cols=None, time_unit='years'):
    """
    Plots a histogram of the difference between two date columns within a specified range.
    
    Parameters:
    - df: DataFrame containing the date columns.
    - date_col1: Name of the first date column.
    - date_col2: Name of the second date column.
    - condition_cols: List of columns for coloring conditions. 
                      Bars corresponding to rows where each condition_col is 1 will be colored differently.
    - time_unit: Unit of time to measure difference ('days', 'months', 'years').
    """
    # Ensure the columns are in datetime format
    df[date_col1] = pd.to_datetime(df[date_col1])
    df[date_col2] = pd.to_datetime(df[date_col2])

    # Calculate the difference between the two dates
    df['DateDifference'] = df[date_col1] - df[date_col2]
    
    # Convert difference to desired time unit
    if time_unit == 'days':
        df['DateDifference'] = df['DateDifference'].dt.days
        number_of_bins = 100  
    elif time_unit == 'months':
        df['DateDifference'] = df['DateDifference'].dt.days / 30.44  
        number_of_bins = 100  
    elif time_unit == 'years':
        df['DateDifference'] = df['DateDifference'].dt.days / 365.25  
        number_of_bins = int(df['DateDifference'].max() - df['DateDifference'].min() + 1)
    else:
        raise ValueError("Invalid time_unit. Choose from 'days', 'months', or 'years'.")
    
    plt.figure(figsize=(10, 6))
    
    # Plot base histogram for all data
    (n, bins, patches) = plt.hist(df['DateDifference'], bins=number_of_bins, edgecolor='black', alpha=0.5, label='All')
    
    # Plot histograms for each condition
    if condition_cols is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(condition_cols)))
        cumulative_condition = np.zeros_like(n)  # Align with histogram bins
        for col, color in zip(condition_cols, colors):
            condition = df[col] == 1
            n_condition, _ = np.histogram(df.loc[condition, 'DateDifference'], bins=bins)
            plt.hist(bins[:-1], bins=bins, weights=n_condition, 
                     color=color, edgecolor='black', alpha=0.5, 
                     bottom=cumulative_condition, 
                     label=col)
            # Update the cumulative_condition if you want the next condition to stack above
            cumulative_condition += n_condition
    
    # Labels and title
    plt.title('Histogram of Payoffs')
    plt.xlabel(f'Years Left on Mortgage ({time_unit})')
    plt.ylabel('Frequency')
    
    # Legend
    plt.legend()
    
    plt.show()