''' NLP Clusters: Give a url of a Yahoo Finance article to get a list of
    similar articles collected since 26 November 2021
    * Input: Article url
    * Output: csv file of similar articles, their sentiments and dates
    * Procedure:
        - Crawl the requested article data
        - Preprocess the text
        - Add NLP i.e. sentiment, keywords, named entities incl. cryptocurrencies
        - 'Ask' the model whether it fits any cluster so far, and if yes,
            which articles are in the cluster
        - Return csv file'''

from model_update import ModelReqest
from webcrawler import Webcrawler
from tkinter import filedialog

def get_raw_df(url):
    webcrawler = Webcrawler()
    url = input('Enter url: ') # Replace this by a custom input procedure
    df = webcrawler.run_single_url(url)
    df = model_request.run_single_document(df)
    cols = ['title', 'text', 'date', 
                'title', 'keywords', 'entities', 
                'sent_title', 'sent_text'] # Columns for export
    df = df[[cols]]
    return(df)

def prep_and_nlp_and_model_request(df):
    model_request = ModelReqest()
    df = model_request.run_single_document(df)
    return(df)

if __name__ == '__main__':
    df = get_raw_df(url)
    df = prep_and_nlp(df)
    path = filedialog.asksaveasfile(filetypes = (("CSV files", "*.csv"), ("All files", "*.*"))).name
    df.to_csv(path)