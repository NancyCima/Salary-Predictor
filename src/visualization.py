import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def boxplots(data, numerical_cols):
    '''
    Creates boxplots for numerical columns in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to visualize.
    numerical_cols : list
        List of column names to visualize.
    '''

    num_vars = len(numerical_cols)
    cols = 3  # Number of graphs per row
    rows = (num_vars + cols - 1) // cols  # Calculation of required rows

    plt.figure(figsize=(6 * cols, 5 * rows))
    plt.suptitle('Distribution of each variable and its mean', fontsize=16, y=1.02)

    for i, v in enumerate(numerical_cols):
        prom = data[v].mean()
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(data=data, x=v, width=0.5)
        plt.axvline(x=prom, color='r', linestyle='--', label='Mean')
        plt.title(v)
        plt.legend()

    plt.tight_layout()
    plt.show()



# Configuramos estilo de seaborn
sns.set_style("darkgrid")

# Function to create subplots and charts
def create_plots(columns, plot_func, n_cols=3):
    """
    Creates a grid of subplots and applies a plotting function to each.

    Parameters
    ----------
    columns : list
        A list of columns to plot.
    plot_func : function
        The function to use to plot the data for each column.
    n_cols : int, optional
        The number of columns in the subplot grid, by default 3.
    """

    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

    # Flatten the axes array for easy iteration, if necessary
    axes = axes.flatten() if isinstance(axes, np.ndarray) and axes.ndim > 1 else axes
    # Check if axes is a single Axes object
    if not isinstance(axes, np.ndarray):  # If axes is a single Axes object, wrap it in a list
        axes = [axes]

    for i, column in enumerate(columns):
        plot_func(column, axes[i])
        axes[i].set_title(f'Distribution of {column}', fontsize=12)
        axes[i].set_xlabel(column, fontsize=10)

    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    # Adjust subplot parameters to prevent overlapping
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust spacing between subplots
    plt.tight_layout(pad=3.0)  # Add padding and try to prevent overlapping
    plt.show()

def scatterplot(df: pd.DataFrame):
  '''
  This function creates a scatterplot matrix for the given DataFrame.
  It plots the relationship between each feature and the target variable (Salary).
  '''
  cols = df.columns.drop(['id', 'Description', 'Job Title', 'Salary'])
  plt.figure(figsize=(12, 8))
  for i, col in enumerate(cols):
    plt.subplot(4, 2, i+1)
    sns.scatterplot(data=df, x=col, y='Salary')
    plt.legend().remove()
  plt.suptitle("Relationship between features y target")
  plt.tight_layout()
  plt.show()