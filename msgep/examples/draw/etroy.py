import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fig, ax = plt.subplots(1, figsize=(12, 8))
# df = pd.read_excel('etroy.xlsx')
# df1 = df.iloc[:, -6]
# # print(df1)
# n = len(df1)
# color = 'b'
# colors = ['#E74C3C'] + ((len(df1) - 1) * ['#F5B7B1'])
# for idx, val in df1.iterrows():
#     plt.plot([val.Year, val.Year],
#              [-20, val.value],
#              color=colors[idx],
#              marker='o',
#              lw=4,
#              markersize=6,
#              markerfacecolor='#E74C3C')
#
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
#
# plt.xlim(0, 1000)
# plt.ylim(0, 3)
# #
# # plt.savefig('chart.png')
df = pd.read_excel('etroy.xlsx')
y = df.iloc[:, -2]
z = df.iloc[:, -6]
# Data
x = df.iloc[:, -1]
# Create a color if the y axis value is equal or greater than 0
my_color = np.where(y >= 0, 'orange', 'skyblue')
your_color = np.where(y >= 0, 'blue', 'skyblue')
# The vertical plot is made using the vline function
plt.vlines(x=x, ymin=0, ymax=y, color=my_color, alpha=0)
plt.scatter(x, y, color=my_color, s=1, alpha=1, label='MS-GEP-I')
plt.vlines(x=x, ymin=0, ymax=z, color=your_color, alpha=0)
plt.scatter(x, z, color=your_color, s=1, alpha=1, label='GEP')
# Add title and axis names
plt.xlabel('Generation')
plt.ylabel('Entropy')
plt.legend(loc='lower right')
plt.savefig('B.png', dpi=400)
# Show the graph
plt.show()
