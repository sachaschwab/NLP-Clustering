import pandas as pd
import numpy as np

import os
import pickle   

import seaborn as sns
from pylab import rcParams
from datetime import datetime
import matplotlib.pyplot as plt
import hdbscan

from pprint import pprint
from string import digits

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

import spacy
from spacy import displacy
from collections import Counter
# Get the following with: python -m spacy download en_core_web_sm
import en_core_web_sm

class ModelReqeust():

    def prep_text(text):
        '''
            Pre-process text pipeline
            Input: A text (string)
            Output: Pre-processed text
        '''
        # Initialise lemmatizer for later use
        lemmatizer = WordNetLemmatizer()
        
        out_text = text.lower()

        # Remove numbers
        remove_digits = str.maketrans('', '', digits)
        out_text = out_text.translate(remove_digits)
        
        # Erase symbols, vary for the case sentence stop 
        # characters needed (e.g. for BERT)
        symbols = "!\"#$%&()*+-.—,/:;<=>?@[\]^‘’_`{|}~[],''\n"
        for i in symbols:
            out_text = out_text.replace(i, ' ')
        # Exclude stop words. Exclude single character words
        splitted_text = out_text.split()
        stopped_text = ""

        for word in splitted_text:
            if word not in stop_words and len(word) > 1:
                # Exclude apostrophe
                word = word.replace('”', '')
                word = word.replace('“', '')
                word = word.replace("'", "")
                # Lemmatize
                word = lemmatizer.lemmatize(word)
                # Compose text. Some words may have 'shrunken' to 
                # one character. Eliminate these.
                if len(word) > 1:
                    stopped_text = stopped_text + ' ' + word

        out_text = stopped_text

        return(out_text.strip())

    def prep_data_column(column):
        '''
            Pre-process text for an entire column
            Input: List (dataframe column) with text
            Output: Pre-processed text
        '''
        prepped_col = []
        for text in column:
            # Check if there is actually text
            if (len(text) > 0):
                prepped_text = prep_text(text)
            else:
                prepped_text = ''
            prepped_col.append(prepped_text)
        return(prepped_col)

    def preprocess_data(file_path):
        ''' 
            Takes a title + text dataframe and preprocesses the text data
            Input: Path to the file that should be preprocessed
            Output: Dataframe with original columns and columns with preprocessed data
            Uses custom Defs: def prep_data_column which usues
                                def prep_text
        '''
        
        df_raw = pd.read_csv(file_path)
        
        # Prepare df to hold the preprocessed data
        columns = ['title', 'text', 'date_time']
        text_cols = ['title', 'text']
        df_prep = pd.DataFrame(columns=columns)

        for col in columns:
            df_prep[col] = df_raw[col]
            
        for col in text_cols:
            prep_col = prep_data_column(df_raw[col])
            # Add preprocessed data 
            df_prep[col + '_prepped'] = prep_col
        # Convert date-time
        df_prep['date'] = df_prep['date_time'].apply(lambda x: datetime.strptime(x, '%B %d, %Y, %I:%M %p').strftime('%Y-%m-%d'))
        return df_prep

    def build_corpus(df):
        ''' 
            Build corpus of titles and articles body (text)
            Input: Dataframe with title and text (body) rows
            Output: Columns having corpus and merged text
        '''
        
        title_corpus = []
        titles_merged = []
        for title in df['title_prepped']:
            spl = title.split()
            title_corpus.append(spl)
            for word in spl:
                titles_merged.append(word)
        fdist_filtered_titles = FreqDist(titles_merged)
        
        text_corpus = []
        text_merged = []
        for text in df['text_prepped']:
            spl = text.split()
            text_corpus.append(spl)
            for word in spl:
                text_merged.append(word)
        
        df['title_corpus'] = title_corpus
        df['text_corpus'] = text_corpus
        df['merged_corpus'] = df['title_prepped'] + df['text_prepped']
        
        return(df)

    def get_tfidf(corpus):
        ''' 
            Build TF-IDF matrix for a given corpus
            Input: Corpus (Preprocessed text strings)
            Output: TF-IDF matrix, the vectorizer object
                (having the labels)
            TODO next version: Limit to TfidfTransformer
        '''
        # Initiate vectorizer
        vectorizer = TfidfVectorizer(analyzer='word')
        # Form the matrix
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Investigate the shape of the matrix
        print('TF-IDF matrix shape: ' + (str(tfidf_matrix.shape)))
        
        return(tfidf_matrix, vectorizer)

    def add_keywords(df, x_keywords, tfidf_matrix, vectorizer):
        ''' 
            Based on TF-IDF, extract x keywords i.e.
                words with highest weights in the order of 
                their appearance in the corpus
            Input: Dataframe of articles, preprocessed text + corpus
            Output: Dataframe with keyword columns
        '''

        # Get TF-IDF matrix (holding the weights) and vectorizer 
        # object (holding the labels)
        names = vectorizer.get_feature_names_out()
        # Convert to operable format
        weights = tfidf_matrix.todense().tolist()
        # Create a dataframe with the results
        weights_df = pd.DataFrame(weights, columns=names)

        # Add empty columns to hold the x keywords and their weights
        df['keywords'] = ''
        df['keyword_weights'] = ''

        # Loop through merged corpus text
        for i in range(0, len(df)):
            text = df['text_corpus'][i]
            vector = []
            # Look up the weight of each word in the TF-IDF 
            # document-weights-per token matrix
            for word in text:
                if word in weights_df:
                    weight = weights_df[word][i]
                    if (word in set(cryptos.Cryptocurrencies.str.lower())):
                        # Setting an extra high weight since this is 
                        # the 'center' of the exercise
                        weight = 0.5
                    vector.append(weight)
                else:
                    vector.append(0.000)

            # Temporary dataframe to hold weights
            data = {'text':text,'weights':vector}
            df_text = pd.DataFrame(data)
            # Find the top x'st weight and remove all 
            # words with a weight below it
            if (len(df_text) >= x_keywords):
                max_x = sorted(vector, reverse=True)[:x_keywords]
                ind_to_drop = []
                for index, item in df_text.iterrows():
                    # Vector for indices where value
                    # is below the top 50 i.e. to drop from temp. df
                    if item['weights'] < max_x[x_keywords-1]:
                        ind_to_drop.append(index)

                df_text = df_text.drop(df_text.index[ind_to_drop])
                df_text = df_text[:x_keywords]

            df['keywords'][i] = df_text['text'].values
            ws = []
            for w in df_text['weights'].values:
                ws.append(w)
            df['keyword_weights'][i] = ws

        return(df, weights_df)

    def get_named_entities(df, weights_df):
        ''' Uses Spacy to identify relevant named entities
            Input: df of raw texts from web crawler, weights_df from TF-IDF
            Output: df with added entities (top 10) and weights
            TODO Next release: Split function into smaller bits
            TODO Next release: Give the cryptocurrencies a higher weight or other mechanism
        '''
        df['entities'] = ''
        df['entities_weight'] = ''
        # Load the spacy dictionary
        nl = en_core_web_sm.load()

        # Provide spacy matches
        for index, row in df.iterrows():
            title = row['title_prepped'] 
            text = row['text_prepped'] 
            full = title + ' ' + text
            full = full.replace("'", ' ')
            full = full.replace(",", ' ')
            full = full.replace("[", ' ')
            full = full.replace("]", ' ')
            doc = nl(full)
            ents = [(X.text, X.label_) for X in doc.ents]
            # Use this for visualisation of results: displacy.render(doc,style="ent",jupyter=True)
            # Filter relevant entities
            rel_ents = [x[0] for x in ents if (x[1]=='PERSON') or (x[1]=='ORG') or (x[1]=='GPE')]
            for cc in cryptos.Cryptocurrencies:
                cc = cc.lower()
                if (cc in full):
                    rel_ents.append(cc)
            # Loop through result and provide weights
            # (i.e. highest weight of individual words in e.g. 'Hillary Clinton')
            # Then, take only those with the 10 highest weights
            f_names = []
            f_weights = []
            for names in rel_ents:
                temp_values = {}
                for name in names.split():
                    name = name.lower()
                    if name in weights_df.columns:
                        temp_values[name] = weights_df[name][index]
                # proceed if name recognised, otherwise skip
                sorted_temp = sorted(temp_values.items(), key=lambda kv: kv[1], reverse=True)
                if len(sorted_temp) > 0:
                    # Store first name and its weight in temporary vectors
                    f_names.append(sorted_temp[0][0])
                    f_weights.append(sorted_temp[0][1])

            # When done, sort the temporary df and leave the top 10
            # Temporary dataframe to hold weights
            data = {'names':f_names,'weights':f_weights}
            df_temp = pd.DataFrame.from_dict(data).drop_duplicates()
            # Sort and delete all non-top 10
            df_temp = df_temp.sort_values(by=['weights'], ascending=False).drop_duplicates()
            # If not 10 in list, fill with "none" (name) and 0 (weight)
            df_null = pd.DataFrame.from_dict({'names':['none'],'weights':[0]})
            if len(df_temp) < 10:
                x = 10 - len(df_temp)
                for i in range(1, x):
                    df_temp = df_temp.append(df_null)
            df_temp = df_temp.head(10)
            df['entities'][index] = list(df_temp['names'])
            df['entities_weight'][index] = list(df_temp['weights'])

        return(df)

    def add_sentiments(df):
        # Prepare df:
        df['sent_title'] = ''
        df['sent_text'] = ''
        # Initiate VADER
        sid = SentimentIntensityAnalyzer()
        # Loop through items and get the sentiment scores
        # The compound is saved in df as relevant value
        for index, item in df.iterrows():
            sent_title = sid.polarity_scores(str(item['title']))
            sent_text = sid.polarity_scores(str(item['text']))
            df.loc[index, 'sent_title'] = sent_title['compound']
            df.loc[index, 'sent_text'] = sent_text['compound']
        return df

    def prepare_for_clustering(df):
        # Extract the numerical value columns as prepared
        exp_df = df[['keyword_weights', 'entities_weight', 'sent_title', 'sent_text']]

        # Spread the arrays into columnar form
        exp = [pd.DataFrame(exp_df[col].tolist()).add_prefix(col) for col in exp_df.columns]
        exp_df = pd.concat([exp[0], exp[1], exp[2], exp[3]], axis= 1)
        exp_df = exp_df.fillna(0)
        # Normalise for output as features for clustering
        features_df = MinMaxScaler().fit_transform(exp_df)
        # Also, create the similarity matrix for model variation (hdbscan)
        sim_matrix = cosine_sim_train = linear_kernel(features_df, features_df)
        return(features_df, sim_matrix)

    def build_model(df, leaf_size):
        # Extract the numerical value columns as prepared
        exp_df = df[['keyword_weights', 'entities_weight', 'sent_title', 'sent_text']]
        # Spread the arrays into columnar form
        exp = [pd.DataFrame(exp_df[col].tolist()).add_prefix(col) for col in exp_df.columns]
        exp_df = pd.concat([exp[0], exp[1], exp[2], exp[3]], axis= 1)
        exp_df = exp_df.fillna(0)
        # Normalise
        features_df = MinMaxScaler().fit_transform(exp_df)
        db = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='manhattan', \
                        algorithm='generic', leaf_size=leaf_size, prediction_data=True).fit(features_df)
        
        return(features_df, db)

    def save_hdbscan_model(model):
        # Serialize hdbscan model
        with open('model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model
    
    def run_single_document(df):
        print('Running NLP jobs...')
        cols = df.columns
        # Preprocess
        for col in cols:
                prep_col = prep_data_column(df[col])
                # Add preprocessed data 
                df[col + '_prepped'] = prep_col
                # Convert date-time
                df['date'] = df['date_time'].apply(lambda x: datetime.strptime(x, '%B %d, %Y, %I:%M %p').strftime('%Y-%m-%d'))
        df = build_corpus(df)
        tfidf_matrix, vectorizer = get_tfidf(df['merged_corpus'] )
        df, weights_df = add_keywords(df, 10, tfidf_matrix, vectorizer)
        df_mod = prepare_for_clustering(df)
        with open('model.pickle', 'rb') as handle: # Adapt the retrieval procedure
            model = pickle.load(handle)
        label, membership_strength = hdbscan.approximate_predict(model, df_mod[0])
        i = 0
        matches = []
        for lab in model.labels_:
            if lab == label[0]:
                matches.append(i)
                i = i + 1
        # TODO: Finalise retrieval of processed data,
        #       append entities, sents and keywords to single document df
        
        


if __name__ == '__main__':
    dir_path = os.path.realpath('data')
    raw_file_name = 'raw_data.csv'
    file_path = dir_path + '/' + raw_file_name
    df = preprocess_data(dataset_path)
    df = build_corpus(df)
    tfidf_matrix, vectorizer = get_tfidf(df['merged_corpus'] )
    df, weights_df = add_keywords(df, 10, tfidf_matrix, vectorizer)
    
    leaf_size = 80  # Parameters for HDBSCAN:# Parameters for HDBSCAN:
    features_df, model = build_model(df, leaf_size)
    save_hdbscan_model(model)