import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Read the whole text.
text = open('text.txt').read()
wordcloud = WordCloud(background_color="white", collocations=False,
                      width=1600, height=800).generate(text)
# Open a plot of the generated image.
svg = wordcloud.to_svg()

open('wordcloud2.svg', 'w').write(svg)

plt.figure( figsize=(10,5), facecolor='k')
plt.imshow(wordcloud, interpolation="bilinear")
plt.savefig('wordcloud.pdf')
#plt.show()