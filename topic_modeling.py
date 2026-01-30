
! pip install gensim
! pip install pyLDAvis

import json
import gensim
import string
import re
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, utils
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import webbrowser
nltk.download('stopwords')
nltk.download('punkt_tab')


def topics(dataset: str, topic_no: int, output_vis: str):

    with open(dataset, "r", encoding="utf-8") as new_file:
        data = json.load(new_file)

    books = []

    for book in data:
        for date, book_text in book.items():
            text_content = book_text
            books.append(text_content)

    preprocessed_texts = []

    # Regular expression to remove punctuations
    pattern = r'[^\w\s]' 

    stop_words = nltk.corpus.stopwords.words('english')

    # Based on the dataset, added stopwords that are common concepts related to publishing, concepts related to format, concepts that are too frequent or generic, concepts that are too specific, and other irrelevant words
    new_stopwords = ['book','books','collection','collections','collective','publication','published','history','histories','social','culture','cultural','space','work','works','text','texts','stories','new','art','visual','archive','archives','archiving','publishing','concept','exhibition','contemporary','museum','research','essay','essays','words','image','form','forms','idea','ideas','graphic','design','poem','poems','poetry','written','writing','drawing','drawings','comics','photos','photographs','photography','photographic','photographers','practice','practices','process','narrative','concept','features','project','series','artist','artists','life','three','print','way','ways','point','various','sometimes','sometime','something','time','times','includes','world','people','includes','alongside','patterns','artistic','years','based','like','s','one','two','first','second','also','presents','many','make','takes','us','zdanevich','zaum','singapore','use','using','jim','year','well','day','offers','part','author','within','days','things','objects','farid','across','zine','others','tree','even','around','looks','beech','together','made','contains','brings','rather','making','table','could','another','aráuz','joey','archie','without','motutapu','ralph','paul','skagit','yet','de','marx','still','may','take','riis','including','genderfail','man','edition','images','issue','often','always','every','see','look','lara','would','find','banda','back','along','since','printed','chalayan','along','want','roma','red','delta','created','sohrab','different','become','found','place','create','must','experience','experiences','featuring','malay','body','explores','behind','much','english','four','nt','linhart','set','anita','alina','kristian','bora','san','msf','mcdonald','songwat','mau','lee','australia','luis','might']
    stop_words.extend(new_stopwords)

    # Lowercase
    for text in books:
        text = text.lower()

    # Tokenize the text using NLTK's word_tokenize()
        tokens = word_tokenize(text)

    # Remove stopwords and numbers, and join words back into a string
        filtered_text = [word for word in tokens if word not in stop_words and not word.isdigit()]
        preprocessed_text = ' '.join(filtered_text)

    # Remove remaining punctuation
        preprocessed_text = re.sub(pattern, '', preprocessed_text)

    # Append preprocessed text to the list
        preprocessed_texts.append(preprocessed_text)

    # Tokenize each string into a list of words
    tokenized_texts = [word_tokenize(text) for text in preprocessed_texts]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    dictionary.save('text_dictionary')

    # LDA model
    lda_model = models.LdaModel(corpus, alpha='auto', num_topics = topic_no, id2word = dictionary)

    # Coherence
    cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()

    for topic_id, topic_words in lda_model.print_topics():
        print(f'Topic {topic_id + 1}: {topic_words}')


    #print(f'Coherence: {coherence}')

    lda_model.save('lda_model')


    # Visualize the topics

    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics = False)
    pyLDAvis.save_html(lda_display, output_vis)
    webbrowser.open(output_vis)

    return lda_model


topics("book_catalog.json", 6, "books_lda_vis.html")


'''
# Check topics of a document

text = 'An oral history of grief and death within queer trans and or black indigenous and people of color communities The idea for this book came from my personal experiences with the death of loved ones I wanted to make space for a deeper inquiry into how queer trans and/or black Indigenous people of colour communities are responding to death navigating grieving and persisting amidst pain and hardship Philippe my pa died in May of 2020 He was a brown man who experienced a lot of suffering and injustices in his life and he died young Whilst I do not want to reduce the aliveness of my pa’s life or diminish his life into his dying I also do not want to soften that he died a premature black and working class death My pa died during the covid19 lockdowns and due to restrictions his last days were with immediate family and friends and his funeral had attendance limits We went into another lockdown in July of 2020 and it was a time of profound solitude in grieving My close friend Alex died in March of 2023 A group of us found out he had overdosed when we were down the coast together on Gadubanud Country Close friends and I organized for loved ones comrades and family to be together the Friday of that week We shared photos memories ate together cried made space for rage which there was a lot of sadness love connection and care Contrasted to what happened when my pa died it was a remarkably different and healing experience for me When grief arrives comprises eleven narratives of tragic deaths ranging from overdose and suicide to murder rare conditions chronic illness and working class death These deaths are often attributed to manifestations of capitalism and its related institutions of violence including white supremacy the prison industrial complex cisheteropatriarchy and colonialism Tragic deaths are different to death that is part of life through aging Death as part of living can be meaningful and dignified whereas tragic deaths are difficult to respond to Reflecting on these tragic deaths prompts us to question which lives society deems worthy of grieving and what defines a grievable life Here on this continent Aboriginal deaths are hidden excluded from public grieving as Indigenous people are dehumanized Similarly in occupied Palestine settler violence dictates who matters and who is disposable Shaped by rich traditions and histories which situate communities as sites of justice and healing the stories inside stand in stark contrast to the prevailing neoliberal individualized and normative responses to grief They are examples of people collaboratively building and sharing resources and envisioning creative approaches to the complexities of grieving When grief arrives also connects with movement histories that actively contribute to ongoing liberation struggles The project utilizes oral histories as a means to narrate stories challenge the idea of expert and create an archive based in community knowledges My hope for these stories is that friends family and communities can offer those experiencing grief with an alternative perspective of themselves and their loved ones who have died rather than just the experiences of the hardships and struggles they are facing When grief arrives took place on the stolen lands of the Wurundjeri & Boon Wurrung/Bunurong peoples of the Kulin Nation Sovereignty was never ceded and this is and always will be Aboriginal land Any profits will be donated to my friends gender affirming feminisation surgery'
lda = models.LdaModel.load('lda_model')
dictionary = Dictionary.load('text_dictionary')
tkn_doc = utils.simple_preprocess(text)
doc_bow = dictionary.doc2bow(tkn_doc)
doc_vec = lda[doc_bow]
print(f'{doc_vec}')
'''
