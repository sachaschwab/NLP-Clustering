#     Author: Sacha Schwab <br>
#     Location: Zurich, Switzerland
#     Date: 8 December 2021
#     MIT licensed

from bs4 import BeautifulSoup
import requests
import datetime
import pandas as pd
import numpy as np
from datetime import date
import os

class Webcrawler():

    def get_page_content(url):
        ''' Request and retrieve html from a webpage, and status code
            Input: the url to be crawled
            Output: A timestamp and status code of the request and the page content
            Prints: The url loaded at the moment, for monitoring purpose
        '''
        print("Getting url: " + url)
        status_codes = {}
        page = requests.get(url)
        status_code = page.status_code
        timestamp = datetime.datetime.now()
        return(status_code, page)

    def get_soup(page):
        ''' Convert the page html content from a request into a beutifulsoup soup
            Input: The page html content
            Output: The soup
        '''
        soup = BeautifulSoup(page.content, 'html.parser')
        return(soup)

    def get_title(soup):
        ''' Extract the title from a Yahoo articles page
            Input: Soup
            Output: The title text
        '''
        # Extract the title
        if soup.find('header', class_='caas-title-wrapper'):
            print('Getting title')
            title = soup.find('header', class_='caas-title-wrapper').text.strip()
            return(title)
        else:
            print('An issue occurred during title crawling.')
            return('')

    def get_date_time(soup):
        ''' Extract the date stamp from a Yahoo articles page
            Input: Soup
            Output: The date text
        '''
        # Extract the date
        if soup.find('div', class_='caas-attr-time-style'):
            date = soup.find('div', class_='caas-attr-time-style').text.split("·")[0]
            return(date)
        else:
            print('An issue occurred during date/time crawling.')
            return('')
    def get_text(soup):
        ''' Extract the body articles text
            Input: Soup
            Output: The article body text
        '''
        print('Crawling text')
        # Extract the article text
        art_text = soup.find('div', class_='caas-body').text
        return(art_text)

    def get_yahoo_crypto_news_only_url(file_path):
        ''' Extract the urls currently feature on Yahoo cryptocurrency news
            Input: n/a
            Output: Urls (i.e. new ones) extracted here are directly save
                    into the raw data file.
        '''
        # Get the soup and the status of the response
        df = pd.read_csv(file_path)
        status, page = get_page_content(yahoo_url)
        soup = get_soup(page)
        # Loop through html items and extract the data
        titles_tags = soup.find_all("a", class_="js-content-viewer", href=True)
        for title_tag in titles_tags:
            url = 'https://yahoo.com' + title_tag['href']
            # Proceed only if the url does not yet exist
            if not (url in df['url']):
                data = {}
                data['url'] = url
                data['title'] = ''
                df = df.append(data, ignore_index=True)
                print('Added and ready for NLP model update: ' + url)
        return(df)

    def crawl_new_articles(file_path):
        ''' Crawl Yahoo articles newly obtained
            Input: Path to the file containing the new urls
            Output: Dataframe with titles and body text data to each new url
            Prints: The url crawled at the moment
        '''
        # Read the raw articles data
        print('Opening raw data file')
        df = pd.read_csv(file_path)
        # Backup just in case
        df.to_csv(dir_path + 'raw_data_backup' + str(date.today()) + '.csv')
        # GOVERNANCE: Clean backups from time to time

        # Erase NaNs
        df = df.fillna('')
        # Filter the urls that have not yet been crawled
        df_todo = df[df['text'] == '']

        # Loop through urls to crawl and get the data
        i = 0
        for index, row in df_todo.iterrows():
            print('Now crawling: ' + row['url'])
            # Dict to hold the sample data
            sample = {}
            # Get response code
            response_code, page = get_page_content(row['url'])
            if response_code == 200:
                # Get the soup
                soup = get_soup(page)
                title = get_title(soup)
                if (len(title) > 0):
                    df.loc[index, 'title'] = title
                    text = get_text(soup)
                    if len(text) > 0:
                        df.loc[index, 'text'] = text
                        df.loc[index, 'date_time'] = get_date_time(soup)
                        
                    else:
                        print('dropping row')
                        df = df.drop(index = index)
                else:
                    df = df.drop(index = index)
            else:
                print('dropping')
                df = df.drop(index = index)
        return df

    def run_single_url(url):
        print('Running webcrawler...')
        status_code, page = get_page_content(url)
        soup = get_soup(page)
        data = {}
        data['title'] = get_title(soup)
        data['text'] = get_text(soup)
        data['date_time'] = get_date_time(soup)
        df = pd.date_range(data=data)
        return(df)
        
if __name__ == '__main__':
    dir_path = os.path.realpath('data')
    raw_file_name = 'raw_data.csv'
    file_path = dir_path + '/' + raw_file_name
    yahoo_url = "https://finance.yahoo.com/cryptocurrencies/"
    df = get_yahoo_crypto_news_only_url(file_path)
    df.to_csv(file_path)
    df = crawl_new_articles(file_path)
    print("Saving crawled data")
    df.to_csv(file_path)