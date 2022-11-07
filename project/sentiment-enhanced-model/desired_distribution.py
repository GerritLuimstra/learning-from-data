import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle

data = np.append(np.random.normal(-0.5, 0.15, 1000), np.random.normal(0.5, 0.2, 600))
n, bins, patches = plt.hist(data, bins=100)

for bin, patch in zip(bins, patches):
    if bin < -0.5:
        patch.set_facecolor(random.choice(["green", "red"]))
        patch.set_alpha(0.5)
    else:
        patch.set_facecolor("green")
        patch.set_alpha(0.5)

plt.title("Hypothesized and ideal sentiment distribution")
plt.xlabel("Sentence Sentiment")
plt.ylabel("Frequency")
plt.yticks([], [])
handles = [Rectangle((0,0), 1, 1, color=c) for c in [(1, 0, 0, 0.5), (0, 1, 0, 0.5)]]
labels= ["Offensive", "Not offensive"]
plt.legend(handles, labels)
plt.show()