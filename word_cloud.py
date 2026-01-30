# Generate work clouds based on the top 50 frequent words of each region

! pip install wordcloud pandas matplotlib

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load file (top 50 frequent words of each region)
df = pd.read_csv("frequency-global south.csv")

# Create a dictionary
word_freq = dict(zip(df["word"], df["frequency"]))

# Generate the word cloud
wc = WordCloud(
    width=600,
    height=400,
    background_color="white"
).generate_from_frequencies(word_freq)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
