import matplotlib.pyplot as plt

fig1, ax = plt.subplots(1, 2)
ax[0].plot(range(10))
ax[1].plot([i * 1.2 for i in range(10)])
ax[0].remove()
ax[1].remove()

fig2 = plt.figure()
ax[0].figure=fig2
ax[1].figure=fig2
fig2.axes.extend(ax)
fig2.add_axes(ax[0])
fig2.add_axes(ax[1])

dummy = fig2.add_subplot(121)
ax[0].set_position(dummy.get_position())
ax[1].set_position(dummy.get_position())
dummy.remove()
dummy.remove()

plt.close(fig1)

plt.show()