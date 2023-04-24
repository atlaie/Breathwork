import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import cmcrameri as cmc
from PIL import Image


# Function to lemmatize words
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_sentences = []
    for sent_words in words:
        lemmatized_words = []
        for word in sent_words:
            pos = get_wordnet_pos(nltk.pos_tag([word])[0][1])
            lemmatized_word = lemmatizer.lemmatize(word, pos)
            lemmatized_words.append(lemmatized_word)
        lemmatized_sentences.append(lemmatized_word)

    return lemmatized_sentences

# Flatten a list of lists [of lists(of lists...)] of arbitrary depth
def flatten(l):
    for i in l:
        if isinstance(i,list):
            yield from flatten(i)
        else:
            yield i

# Data cleaning and preprocessing
def preprocess(texts):
    preprocessed_texts = []
    for text in texts:
        # Remove unnecessary characters and convert to lowercase
        cleaned_text = text.lower().replace('\n', ' ')
        
        # Tokenize sentences and words
        sentences = sent_tokenize(cleaned_text)

        # Tokenize words and remove punctuation
        words = [nltk.word_tokenize(sent) for sent in sentences]
        words = [[w.lower() for w in sent_words if w.isalnum()] for sent_words in words]

        # Remove English stopwords
        stop_words = set(stopwords.words('english'))
        words = [[w for w in sent_words if w not in stop_words] for sent_words in words]
        
        preprocessed_texts.append({"text": cleaned_text, "sentences": sentences, "words": words})
    
    return pd.DataFrame(preprocessed_texts)


# Basic statistics and visualization
def basic_statistics(df):
    num_documents = len(df)
    total_sentences = df['sentences'].apply(len).sum()
    total_words = df['words'].apply(lambda x: sum(len(sent) for sent in x)).sum()
    unique_words = len(set(word for doc_words in df['words'] for sent_words in doc_words for word in sent_words))
    
    print("Number of documents:", num_documents)
    print("Total sentences:", total_sentences)
    print("Average sentences per document:", total_sentences / num_documents)
    print("Total words:", total_words)
    print("Average words per sentence:", total_words / total_sentences)
    print("Unique words:", unique_words)


def visualize_length_distribution(df):
    sentence_lengths = df['sentences'].apply(len)
    word_lengths = df['words'].apply(lambda x: sum(len(sent) for sent in x))
    
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.violinplot(data = sentence_lengths)
    plt.title("Sentence Length Distribution")
    plt.xlabel("")
    
    plt.subplot(1, 2, 2)
    sns.violinplot(data = word_lengths)
    plt.title("Word Length Distribution")
    plt.xlabel("")

    sns.despine()

    plt.show()


def generate_word_cloud(df):
    all_words = ' '.join(list(flatten(df['stemmed_words'].tolist())))
    #Create the mask
    breathing_mask = np.array(Image.open('face-exhaling.png'))

    wordcloud = WordCloud(mask = breathing_mask, background_color='white', colormap = 'cmc.turku_r', max_words = 100,collocations=True,
                      contour_color='black',
                      contour_width=1).generate(all_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    sns.despine()
    # plt.savefig('EmojiFace_WordCloud.png', transparent = True, dpi = 200)
    plt.show()

def visualize_pos_distribution(df):
    pos_tags = [nltk.pos_tag(sent_words) for doc_words in df['words'] for sent_words in doc_words]
    pos_counts = nltk.FreqDist(tag for (_, tag) in [tagged_word for sent_tags in pos_tags for tagged_word in sent_tags])
    pos_df = pd.DataFrame(pos_counts.most_common(), columns=["POS", "Count"])

    plt.figure(figsize=(12, 6))
    sns.barplot(x="POS", y="Count", data=pos_df, palette="Greys_r")
    plt.xticks(ticks = np.arange(pos_df.shape[0]), labels = pos_df['POS'].values, rotation = 30)
    plt.title("POS Distribution")
    plt.xlabel('')
    sns.despine()
    plt.show()

def plot_top_bigrams(df, nrows=4, ncols=4):
    # Set plotting style and font scale
    sns.set(style="whitegrid", font_scale=1.1)
    
    # Set up the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    fig.tight_layout(pad=6)

    # Iterate through each text and plot the top bigrams
    for i, row in df.iterrows():
        ax = axes[i // ncols][i % ncols]
        
        # Get the top bigrams and their frequencies
        bigram_measures = BigramAssocMeasures()
        bigram_finder = BigramCollocationFinder.from_words(flatten(row['words']))
        bigram_finder.apply_freq_filter(1)
        top_bigrams = bigram_finder.nbest(bigram_measures.raw_freq, 5)
        bigram_freqs = [bigram_finder.ngram_fd[bigram] for bigram in top_bigrams]

        # Prepare bigrams labels as strings for plotting
        bigram_labels = [f"{bigram[0]} {bigram[1]}" for bigram in top_bigrams]

        # Plot the top bigrams
        sns.barplot(x=bigram_labels, y=bigram_freqs, ax=ax, palette="Greys_r")
        ax.set_title(f"Top 5 Bigrams for Subject {i+1}")
        ax.set_xticks(ax.get_xticks()+0.5)
        ax.set_xticklabels(bigram_labels, rotation=45, ha="right")
        if i in [0, 4, 8, 12]:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', colors=(0,0,0,0))
        ax.set_ylim([0, 5])
    # Adjust the layout to avoid overlapping titles and labels
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    sns.despine()
    plt.show()

# Modified function to plot word clouds with stemmed words
def plot_stemmed_word_clouds(df, nrows=4, ncols=4):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

    for i, row in df.iterrows():
        ax = axes[i // ncols][i % ncols]
        
        # Generate word cloud
        wc = WordCloud(width=400, height=400, stopwords = set(), colormap = 'cmc.turku_r', max_words = 25, background_color="white", random_state=42)
        text = ' '.join(flatten(row['stemmed_words']))
        wc.generate(text)

        # Plot the word cloud
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Stemmed Word Cloud for Subject {i+1}")

    plt.tight_layout(pad=2)
    plt.show()

