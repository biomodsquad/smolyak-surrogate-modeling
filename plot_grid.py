import smolyay
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use(['mphowardlab','aip'])
import numpy
# slow smolyak grid
size = 15
linewidth = 0
samples = [smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 4),smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3)]
grid = smolyay.samples.SmolyakSparseProductPointSet(samples)
gridpoints = grid.points
fig, ax = plt.subplots(1,1,subplot_kw=dict(box_aspect=1),figsize=(4,4))
ax.scatter(*numpy.array(gridpoints).T,linewidth=linewidth,s=size)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('mixed_smolyak_grid.png',bbox_inches='tight')
plt.close()
