import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
 
# Sample data
data = {
    'Number of Inference Steps': [0, 20, 50, 100, 200, 500],
    'MiDa': [1107, 1104, 1106, 1095, 1099, 1101],
    'Mida w/o DA': [1060, 1063, 1058, 1021, 1004, 989],
    'DiffRec': [1058, 1032, 1020, 971, 938, 909]
}
df = pd.DataFrame(data)
# sns.set_style("dark")
 
# Set the style and font sizes
# sns.set_style('ticks')
plt.rcParams.update({'font.size': 8})
 
# Create a Seaborn line plot with different markers for each product
sns.lineplot(x='Number of Inference Steps', y='MiDa', data=df, marker='o', color='navy', label='MiDa')
sns.lineplot(x='Number of Inference Steps', y='Mida w/o DA', data=df, marker='s', color='salmon', label='MiDa w/o DA')
g = sns.lineplot(x='Number of Inference Steps', y='DiffRec', data=df, marker="X", color='seagreen', label='DiffRec')
# [0, 10, 20, 50, 100, 200, 500]
# g.set_xticklabels([0, 10, 20, 50, 100, 200, 500])
# Set plot title and axes labels
plt.title('Inference Steps Impact at ML-1M')
plt.xlabel('Number of Inference Steps')
plt.ylabel('Recall@10 (*1e3)')
plt.xticks([0, 20, 50, 100, 200, 500])
# Add a legend
plt.legend(loc='lower right')
 
 
# Remove the top and right spines
# sns.despine()
# sns.despine(left=True)

# Show the plot
# plt.show()
plt.savefig('sampleParam.png')