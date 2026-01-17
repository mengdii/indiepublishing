

! pip install wordcloud pandas matplotlib

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("frequency-global south.csv")

# Create a dictionary: {word: frequency}
word_freq = dict(zip(df["word"], df["frequency"]))

# Generate the word cloud
wc = WordCloud(
    width=600,
    height=400,
    background_color="white"
).generate_from_frequencies(word_freq)

# Display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()