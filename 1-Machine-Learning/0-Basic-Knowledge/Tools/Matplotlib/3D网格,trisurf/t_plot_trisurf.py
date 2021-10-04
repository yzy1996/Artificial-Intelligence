import numpy as np
import pickle

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.tri as mtri
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.style.use('default')

np.set_printoptions(threshold=np.inf)

with open('pml_predicted_pf.pickle', 'rb') as f:
    predicted_pf, evaluated_points = pickle.load(f)

x, y, z = predicted_pf[:, 0], predicted_pf[:, 1], predicted_pf[:, 2]

# 计算三角形索引 [17158, 3]
tri = mtri.Triangulation(x, y).triangles

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = colors.ListedColormap(['Tomato'])

# ss = 7000
# color = np.arange(ss)
# ax.scatter(x[:ss], y[:ss], z[:ss], s = 1, c=color)

# # 计算坐标距离
xx = x[tri]
yy = y[tri]
zz = z[tri]

mask = []
threshold = 0.15

for i in range(len(tri)):
    m = np.vstack((xx[i], yy[i], zz[i])).T
    pose1, pose2, pose3 = m[0], m[1], m[2]
    dist1 = np.linalg.norm(pose1 - pose2)
    dist2 = np.linalg.norm(pose2 - pose3)
    dist3 = np.linalg.norm(pose3 - pose1)

    dist = dist1 + dist2 + dist3

    if dist <= threshold:
        mask.append(0)
    elif dist <= 0.3 and min(tri[i]) >= 1000:
        mask.append(0)
    elif dist <= 0.5 and min(tri[i]) >= 9000:
        mask.append(0)
    else:
        mask.append(1)
    # with open('2.txt', 'a') as f:
    #     f.write(str(dist))
    #     f.write('\n')


triang = mtri.Triangulation(x, y)
triang.set_mask(mask)


ax.plot_trisurf(triang, z, cmap=cmap, alpha=0.2,
                linewidth=0.2, antialiased=False)
ax.scatter(evaluated_points[:, 0], evaluated_points[:, 1], evaluated_points[:,
           2], c='C1', alpha=1, s=10, label='PML Non-Dominated Sols.', zorder=2)

max_lim = np.max(predicted_pf, axis=0)
min_lim = np.min(predicted_pf, axis=0)

ax.set_xlim(min_lim[0], 1.1 * max_lim[0])
ax.set_ylim(1.1 * max_lim[1], min_lim[1])
ax.set_zlim(min_lim[2], 1.1 * max_lim[2])

ax.set_xlabel(r'$f_1(x)$', size=12)
ax.set_ylabel(r'$f_2(x)$', size=12)
ax.set_zlabel(r'$f_3(x)$', size=12)


# handle the legend
handles = []
pareto_front_label = mpatches.Patch(
    color='tomato', alpha=0.5, label='Approximate Pareto front')
eval_points_label = plt.Line2D(
    (0, 1), (0, 0), color='C1', marker='o', linestyle='', label='PML Non-Dominated Sols')

handles.extend([pareto_front_label])
handles.extend([eval_points_label])

ax.legend(handles=handles, bbox_to_anchor=(1, 1))

plt.show()
fig.savefig('pml_pred.pdf', format='pdf', dpi=1200, bbox_inches='tight')
