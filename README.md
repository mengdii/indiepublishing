This repository is for the data and code of my MA thesis: Political resistance in independent publishing: analyzing independent publications using topic modeling and clustering. The methodological framework is situated in digital humanities approaches, using text analysis, clustering, and topic modeling to examine a dataset of over 1000 independent publications. 

### Research questions
1. What are the common social and political themes among independent publications? What are the differences among geopolitical regions?
2. How does independent publishing function as a form of resistance? Which practices do independent publishers use? And how do independent publishers identify their roles in social change?

### Findings
The findings suggest that while there is an overarching theme of using life experiences to discuss social issues rooted in history, themes of independent publications differ among publishers from the Global South and the Global North, which is affected by their social and political contexts. In terms of publishing practices, a large number of independent publishers tend to use subtle and implicit approaches as everyday resistance instead of radical actions. They imply power dynamics behind everyday life, tell personal stories, respond to social crises, and advocate for the marginalized.

### Data
Independent publications.csv and Independent publishers.csv:
The main datasets comprise content summaries of publications and self-introductions of publishers, which were manually collected from the websites of independent publishers, PDF catalogs from independent publishers, and Instagram posts by independent publishers. The dataset currently contains 1105 publications from 182 publishers. 

### Methods
The methods used in this thesis include text analysis, clustering and topic modeling. During preprocessing, words in British English are converted to American English in unify_to_american_english.py. For topic modeling, the content summaries of all publications are stored in a .json file, where each publication is stored in a dictionary (in text_to_json.py). Other preprocessing steps for topic modeling include lowercasing, removing stopwords, numbers and punctuations, and tokenizing, which is done in topic_modeling.py.

word_cloud.py:
Word frequencies are visualized based on the top 50 frequent words of each region.

keyword_frequency.py:
Keyword frequencies are examined over time from 2015 to 2025 to understand how independent publications respond to major social and political events.

clustering.py:
To identify connections between publications and detect potential themes, document embeddings are used to capture the semantic meanings behind the publications, primarily following the tutorial [Understanding, Generating, and Visualizing Embeddings](https://www.dataquest.io/blog/understanding-generating-and-visualizing-embeddings/). When transforming the publication content summaries into embeddings, a pre-trained model from the [sentence-transformers library](https://huggingface.co/sentence-transformers) all-MiniLM-L6-v2 is used. Clusters are generated with K-Means. The clustering process was done multiple times to find the optimal number of clusters. Dimensionality reduction is used to visualize high-dimensional embeddings in 2D.  To explore and compare regional thematic patterns, scatter plots showing clusters of publications from each region are created and compared.

topic_modeling.py:
To find out publishers’ practices from the publications, the LDA model in [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html) is used to conduct topic modeling. The number of topics is set to 6 to balance coherence, interpretability and the ability to summarize the thematic structure of the dataset. The Python library for interactive topic model visualization [pyLDAvis](https://pypi.org/project/pyLDAvis/) is used to visualize the results. To interpret the themes, the top 30 most relevant terms for each topic are examined, applying close reading on the original text with social contexts taken into account.
