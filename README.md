This repository is for the data and code for my thesis Political resistance in independent publishing: analyzing independent publications using topic modeling and clustering. My research is based on the theories of alternative media and everyday resistance or infrapolitics. The methodological framework is situated in digital humanities approaches, using text analysis, clustering, and topic modeling to examine a self-compiled dataset of over 1000 independent publications. The main research questions are: 1. What are the common social and political themes among independent publications? 2. How does independent publishing function as a form of resistance? Which approaches do independent publishers take?

The findings suggest that while there is an overarching theme of using life experiences to discuss social issues rooted in history, themes of independent publications differ among publishers from the Global South and the Global North, which is affected by their social and political context. In terms of publishing practices, a large number of independent publishers tend to use subtle and implicit approaches as everyday resistance instead of radical actions. They imply power dynamics behind everyday life, tell personal stories, respond to social crises, and advocate for the marginalized.

Independent publications.csv and Independent publishers.csv:
The main datasets contain the content summaries of publications and self-introduction of the publishers, which were manually collected through websites of independent publishers, PDF catalogs from independent publishers, and Instagram posts by independent publishers.

The methods used in this thesis include text analysis, clustering and topic modeling. During preprocessing, words in British English are converted to American English in unify_to_american_english.py. For topic modeling, the content summaries of all publications are stored in a .json file, where each publication is stored in a dictionary (in text_to_json.py). Other preprocessing steps for topic modeling include lowercasing, removing stopwords, numbers and punctuations, and tokenizing, which is done in topic modeling.py.

word_cloud.py:
Word frequencies are visualized based on the top 50 frequent words of each region.

keyword_frequency.py:
Keyword frequencies are examined over time from 2015 to 2025 to understand how independent publications respond to major social and political events.

clustering.py:
To identify connections between publications and detect potential themes, document embeddings are used to capture the semantic meanings behind the publications, primarily following the tutorial Understanding, Generating, and Visualizing Embeddings (M. Levy, 2025). When transforming the publication content summaries into embeddings, a pre-trained model from the sentence-transformers library all-MiniLM-L6-v2 (Reimers et al., n.d.) is used. Clusters are generated with K-Means. The clustering process was done multiple times to find the optimal number of clusters. Dimensionality reduction is used to visualize high-dimensional embeddings in 2D.  To explore and compare regional thematic patterns, scatter plots showing clusters of publications from each region are created and compared.

topic_modeling.py:
To find out publishers’ practices from the publications, the LDA model in Gensim (Řehůřek, 2024) is used to conduct topic modeling. The training step was conducted alongside the data collection process. The training dataset was run with the number of topics set to 5 to 15. Eventually the number of topics is set to 6 to balance coherence, interpretability and the ability of summarizing the thematic structure of the dataset. The Python library for interactive topic model visualization (Mabey, 2023) is used to visualize the results. To interpret the themes, the top 30 most relevant terms for each topic are examined, applying close reading on the original text with socio-historical context taken into account.
