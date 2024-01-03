import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = [[5, 15, 20, 18],
        [14, 18, 22, 20],
        [18, 30, 32, 28]]

rows = ['ML-1M', 'Yelp', 'Amazon-book']
colums = ['MiDa', 'MVAE', 'MDAE', 'DiffRecs']
heads = ['Dataset', 'Method', 'Number of parameters (M)']
lst = []
for ii in range(len(data)):
	for jj in range(len(data[ii])):
		lst.append([rows[ii], colums[jj], data[ii][jj]])

print(lst)
df = pd.DataFrame(lst, columns = heads)
# # Convert the data into a DataFrame for Seaborn

# # Set the style of seaborn
# sns.set(style="whitegrid")

# # Create a bar chart using Seaborn
sns.barplot(x = "Dataset", y = "Number of parameters (M)", hue = "Method",  data=df, palette="viridis")
# # Set labels and title
# plt.xlabel('Groups')
# plt.ylabel('Values')
# plt.title('Bar Chart for 3 Groups of 4 Values')

# # Show the plot
# plt.show()
plt.savefig('sampleParam.png')
