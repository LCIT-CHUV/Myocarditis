# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:25:40 2024

Adapted from : https://tosinharold.medium.com/enhancing-correlation-matrix-heatmap-plots-with-p-values-in-python-41bac6a7fd77
Adapted from : Tosin Harold Akingbemisilu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau


df_p=pd.read_excel("pval_heatmap_myo.xlsx",index_col=[0])
df_r=pd.read_excel("spearR_val_heatmap_myo.xlsx",index_col=[0])


# Generate the correlation matrix afresh
corr = df_r
p_values = df_p

# mask the correlation matrix to diagonal
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask, False)

fix,ax = plt.subplots(figsize=(80,40))
plt.title("Correlation map with P-value", fontsize=14)

# Generate heatmap
heatmap = sns.heatmap(corr,
                      annot= True,
                      annot_kws={"fontsize": 20},
                      fmt='.2f',
                      linewidths=0.5,
                      cmap='RdBu',
                      mask=mask,
                      ax=ax)

colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=25)

# Diag mask and min max (for font color)
mask_pvalues = np.triu(np.ones_like(p_values), k=1)
max_corr = np.max(corr.max())
min_corr = np.min(corr.min())


for i in range (p_values.shape[0]):
  for j in range(p_values.shape[1]):
    if mask_pvalues[i, j]:
      p_value = p_values.iloc[i, j]
      if not np.isnan(p_value):
        correlation_value = corr.iloc[i, j]
        text_color = 'white' if correlation_value >= (max_corr - 0.4) or correlation_value <= (min_corr + 0.4) else 'black'
        if p_value <= 0.001:
            #include double asterisks for p-value <= 0.01
            ax.text(i + 0.5, j + 0.8, f'p = {p_value:.1e}***',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    color=text_color, weight="bold")
        elif p_value <= 0.01:
            #include double asterisks for p-value <= 0.01
            ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})**',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    color=text_color, weight="bold")
        elif p_value <= 0.05:
            #include single asterisk for p-value <= 0.05
            ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})*',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    color=text_color, weight="bold")
        else:
            ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.2f})',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    color=text_color)

# Customize x-axis labels
x_labels = [textwrap.fill(label.get_text(), 13) for label in ax.get_xticklabels()]
ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=20)

# Customize y-axis labels
y_labels = [textwrap.fill(label.get_text(), 13) for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=25)

# Display the plot
plt.show()