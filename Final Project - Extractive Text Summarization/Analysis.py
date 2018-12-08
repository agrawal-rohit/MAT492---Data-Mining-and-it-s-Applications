import numpy as np
import matplotlib.pyplot as plt

# Rouge-1 plot
N = 5
basic_freq = (0.1446, 0.07855320103569793, 0.07149670679082444, 0.12787114845938377, 0.11082453494218199)
complex_tfidf = (0.1115, 0.10255374743549546, 0.04913728369950779, 0.15639182522903453, 0.16607366404087714)
textrank = (0.0862, 0.1422820688083846, 0.09631868131868129, 0.361917502787068, 0.2858104858104858)
lsa = (0.1152, 0.08583251873574453, 0.025354918151528322, 0.15933707848601464, 0.15865311455886733)

ind = np.arange(N)  # the x locations for the groups
width = 0.2     # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, basic_freq, width, color='b')
rects2 = ax.bar(ind+width, complex_tfidf, width, color='r')
rects3 = ax.bar(ind+width+width, textrank, width, color='m')
rects4 = ax.bar(ind+width+width+width, lsa, width, color='c')

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_xlabel('Document')
ax.set_title('Rouge-1 Scores for each technique used')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('D1', 'D2', 'D3', 'D4', 'D5') )

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Basic Tern Frequency', 'Complex TF-IDF','TextRank', 'Latent Semantic Analysis') )
plt.show()

# Rouge-2 plot
N = 5
basic_freq = (0.0315, 0.0, 0.0, 0.06166666666666667, 0.010526315789473684)
complex_tfidf = (0.0172, 0.0, 0.0, 0.05980497341630307, 0.03852201257861635)
textrank = (0.0, 0.0, 0.0, 0.1533133533133533, 0.052000000000000005)
lsa = (0.0272, 0.0, 0.0, 0.05336765336765337, 0.011428571428571429)

ind = np.arange(N)  # the x locations for the groups
width = 0.2     # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, basic_freq, width, color='b')
rects2 = ax.bar(ind+width, complex_tfidf, width, color='r')
rects3 = ax.bar(ind+width+width, textrank, width, color='m')
rects4 = ax.bar(ind+width+width+width, lsa, width, color='c')

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_xlabel('Document')
ax.set_title('Rouge-2 Scores for each technique used')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('D1', 'D2', 'D3', 'D4', 'D5') )

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Basic Tern Frequency', 'Complex TF-IDF','TextRank', 'Latent Semantic Analysis') )
plt.show()

# Bleu plot
N = 5
basic_freq = (0.1446, 0.19122596104341577, 0.22491628847334486, 0.1701504081247597, 0.2558826482468122)
complex_tfidf = (0.1115, 0.4072510077438178, 0.08822581797290707, 0.18800714790861797, 0.3006928883317841)
textrank = (0.0862, 0.3830603780636676, 0.2190212135577319, 0.3179855248910225, 0.3124910400159282)
lsa = (0.1152, 0.16986249232620293, 0.053603940426276836, 0.1351871415839707, 0.10498652047143889)

ind = np.arange(N)  # the x locations for the groups
width = 0.2     # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, basic_freq, width, color='b')
rects2 = ax.bar(ind+width, complex_tfidf, width, color='r')
rects3 = ax.bar(ind+width+width, textrank, width, color='m')
rects4 = ax.bar(ind+width+width+width, lsa, width, color='c')

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_xlabel('Document')
ax.set_title('Bleu Scores for each technique used')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('D1', 'D2', 'D3', 'D4', 'D5') )

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Basic Tern Frequency', 'Complex TF-IDF','TextRank', 'Latent Semantic Analysis') )
plt.show()