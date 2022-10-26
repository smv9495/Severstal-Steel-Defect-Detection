import numpy as np
import matplotlib.pyplot as plt
import os

fig, axs = plt.subplots(1,1,figsize=(6,2))
palette = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
img = np.array(palette).reshape((1,4,3))
axs.xaxis.tick_top()
axs.set_xticks([0,1,2,3], ['Defect 1', 'Defect 2', 'Defect 3', 'Defect 4'], fontsize=14)
axs.set_yticks([], [])
axs.imshow(img)
fig.savefig(os.path.join('static', 'palette.png'))
plt.show()