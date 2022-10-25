import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('partbest.xlsx')
# 5MS-GEP-A 3 NMO-SARA,6MS-GEP-I 4GEP,7MS-GEP 2FF-GEP
y = df.iloc[:, -7]
z = df.iloc[:, -2]
# Data
x = df.iloc[:, -1]
# Create a color if the y axis value is equal or greater than 0
# The vertical plot is made using the vline function
plt.hlines(1000, 0, 1000, colors='k', linestyles=':')
# plt.vlines(300, 400, 1000, colors='green', linestyles=':', label='300 generations')
plt.plot(x, y, color='red', alpha=0.7, label='MS-GEP')
plt.plot(x, z, color='blue', alpha=0.7, label='FF-GEP')
# Add title and axis names
plt.xlabel('Generation')
plt.ylabel('fitness')
plt.legend(loc='lower right')
plt.savefig('three.png', dpi=500)
# Show the graph
plt.show()


