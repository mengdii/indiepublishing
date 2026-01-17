

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

    # Regular expression pattern to remove punctuations
    pattern = r'[^\w\s]' # Matches any character that is not an alphanumeric character or whitespace

    stop_words = nltk.corpus.stopwords.words('english')

    # Based on the dataset, added stopwords that are either too frequent or less meaningful
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

    # Remove remaining punctuation marks using regular expressions
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
    # print(f'Number of publications: {len(books)}')

    return lda_model



topics("book_catalog.json", 6, "books_lda_vis.html")



# Check topics of a document

"""
text = '51 Tan is a commissioned record of the Shanghai street vendors that Cao Feile took notice of and made observation of from 2016 to 2023 Initially published as a series of social media posts on the 11th Shanghai Biennale WeChat platform under 51 Personae Project in the form of Lianhuanhua or Chinese pulp comics the record outlines a collection of itinerant peddlers she took efforts to approach get close to and interact with The details are vivid subtle and warmly human the way the peddlers decorate their places the logic with which they make construction the materials they use the spatial strategies they develop and the way they communicate with pedestrians through these strategies The fading street life businesses and vitality of Shanghai began to show signs in early 2016 Starting from 2015 and peaking in 2016 the demolition of illegal constructions movement was changing the familiar streets of ordinary people and the small shops and vendors upon which daily life relied leaving people astonished and bewildered Invited by 51 Personae Project Feile a female architect was invited to document in the form of observation based documentary comics and released the 51 Tan Stalls series on the WeChat of the Power Station of Ar This invitation turned her into a street recorder squatting on the streets learning the visible ways of resolution and ways of coping in street survival Three years later in 2019 the 39 stalls recorded between 2016 and 2017 were compiled into a small book omitting specific locations and characters leaving only their life forms Although the climax of the demolition of illegal constructions had passed by then the disappearance of street vendors roadside markets and the non-standard but flexible and human street commerce was a fundamental fact in everyone memory The disappearance of the stalls also took away pedestrians perception of the streets their rights and their awareness of rights In the historical process of shaping Shanghai streets and roads have always been able to become temporary and seasonal living spaces and have far exceeded their transportation functions constituting an important part of public life With thirty years of continuous demolition the stigmatization of the so called messiness and arbitrary changes to the boundaries of violations the streets gradually lost human life Autonomous and creative street life has been constantly suppressed and deprived and urban life has gradually retreated to life within walls within the walls of commercial housing and shopping malls Since its first publication in 2019 51 Tan has been loved by readers Therefore this project did not stop after the end of the current Shanghai Biennale and the continuous enrichment of content came from enthusiastic submissions and the continued observation and collection by the original authors Until October 2023 all 51 stalls were finally completed Some of the stalls and people in this book are still there some have moved to another place due to intense urban changes some have closed and some have completely left At the same time there are new stalls temporary or long term pretended or sincere organized or unorganized vendors constantly emerging'

lda = models.LdaModel.load('lda_model')
dictionary = Dictionary.load('text_dictionary')
tkn_doc = utils.simple_preprocess(text)
doc_bow = dictionary.doc2bow(tkn_doc)
doc_vec = lda[doc_bow]
print(f'{doc_vec}')
"""