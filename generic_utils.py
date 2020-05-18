import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import re
from textwrap import wrap
import itertools

def do_confusion_matrices(actual, predicted, normalize = False, colormap = 'Blues', labels = None):
	""" For plotting nice confusion matrices """
	cm = 1.*confusion_matrix(actual, predicted, labels=None)
	if normalize:
		for ii in range(cm.shape[0]):
			cm[ii, :] = cm[ii, :]/cm[ii, :].sum()
	np.set_printoptions(precision=2)

	fig = plt.Figure(figsize=(3, 3), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)
	im = ax.imshow(cm, cmap=colormap)
	if labels:
		classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
				for x in labels]
		classes = ['\n'.join(wrap(l, 40)) for l in classes]
		tick_marks = np.arange(len(classes))

		ax.set_xlabel('Predicted', fontsize=7)
		ax.set_xticks(tick_marks)
		c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
		ax.xaxis.set_label_position('bottom')
		ax.xaxis.tick_bottom()

	ax.set_ylabel('True Label', fontsize=7)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize=4, va='center')
	ax.yaxis.set_label_position('left')
	ax.yaxis.tick_left()
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, '%.2g' % cm[i,j],
				horizontalalignment="center", fontsize=6, verticalalignment='center', color="black")

	fig.colorbar(im, ax=ax)
	fig.set_tight_layout(True)

	b, t = ax.get_ylim()  # discover the values for bottom and top
	b += 0.5  # Add 0.5 to the bottom
	t -= 0.5  # Subtract 0.5 from the top
	ax.set_ylim(b, t)  # update the ylim(bottom, top) values
	plt.show()
	return fig