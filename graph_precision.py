import matplotlib.pyplot as plt
import numpy as np

accuracies = [36.75, 36.36]
precisions = [(1.00, 0.00), (0.26, 0.25), (0.00, 0.00), (0.70, 0.80), (0.00, 0.00)]
instruments = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']
precisions2 = [1.00, 0.00, 0.26, 0.25, 0.00, 0.00, 0.70, 0.80, 0.80, 0.00, 0.00]

train_prec = np.array([1.00, 0.26, 0.00, 0.70, 0.00])*100
test_prec = np.array([0.00, 0.25, 0.00, 0.80, 0.00])*100
indices = np.arange(5)


#plt.figure()

fig, ax = plt.subplots()


bar_width = 0.3

rects1 = ax.bar(indices, train_prec, bar_width, color='b', label='Training Accuracy '+str(accuracies[0])+'%')
rects2 = ax.bar(indices+bar_width, test_prec, bar_width, color='y', label='Testing Accuracy '+str(accuracies[1])+'%')

#for i, value in enumerate(precisions2):
#    ax.bar(i, value, bar_width)
#    ax.text(value, i, str(value), color='blue', fontweight='bold')


ax.set_title('Train/Test Precision by Instrument')
ax.set_ylabel('Precision (%)')
ax.set_xlabel('Instrument')
ax.set_xticks(indices+bar_width/2)
ax.set_xticklabels(instruments)
ax.legend()
fig.tight_layout()
plt.show()

