# NLP-Clustering
### Web crawling and NLP engines and clustering of same-event news articles

### Assessment 3 of MA5851 (Data Science Master Class 1) at James Cook University

Author: Sacha Schwab
MIT license

## Quick outline
<ul style="line-height: 1.5; font-size:12pt">
  <li>Crawl Yahoo Finance Cryptocurrency news articles</li>
  <li>Raw text data is preprocessed, embedded (TF-IDF)</li>
  <li>NLP engine runs keyword extraction based on TF-IDF weights, named entity extraction and sentiment analysis</li>
  <li>HDBSCAN algorithm used for clustering, with currently moderate effectiveness (to be enhance in upcoming versions).</li>
</ul>
See architecture outline at the bottom of this page.

## For class Tutors
<ul style="line-height: 1.5; font-size:12pt">
  <li>Reports are in /main as 'A3_DocumentNumber_X_sacha_schwab<' as per assessment outline/li>
  <li>Code files: (1) 'code_webcralwer.ipynb', (2) 'code_nlp.ipynb'</li>
  <li>Model available under /main/model</li>
   <li>For privacy reasons the audio annotated Powerpoint presentation is not available here but in the assessment folder in JCU Learn</li>
</ul>

## Base requirements
<ul style="line-height: 1.5; font-size:12pt">
  <li>Git</li>
  <li>Python 3.7</li>
  <li>Any IDE supporting Jupyter Notebook files</li>
</ul>

## Deploy
<ul style="line-height: 1.5; font-size:12pt">
  <li>Schedule daily_jobs/webcrawler.py code for daily run (ipynb version is for grading)</li>
  <li>Schedule daily_jobs/model_update.py for daily run</li>
  <li>TBD: Get connected articles to a new article by running get_cluster from model_run.py</li>
</ul>

## Architecture
<br>
    
<img width="1173" alt="architecture_" src="https://user-images.githubusercontent.com/10763939/145041976-08a6ccf0-1fb9-4290-90aa-de5607582a91.png">
    
    
    ![crawler_demo](https://user-images.githubusercontent.com/10763939/145158440-65b62795-18df-4990-8d91-319837bb05b0.gif)

    <img width="1004" alt="cluster_display__" src="https://user-images.githubusercontent.com/10763939/145158416-d60070e0-07b4-4b1a-b2fb-4e55f31d6a0c.png">

    <img width="1163" alt="daily_crawler_screenshot" src="https://user-images.githubusercontent.com/10763939/145158471-eed1190b-df1f-4ade-bdf5-815c541d43db.png">

<img width="1429" alt="article_page_screenshot" src="https://user-images.githubusercontent.com/10763939/145158400-e7adf48d-341c-476d-92ff-09b2bf59069a.png">

<img width="1163" alt="daily_crawler_screenshot" src="https://user-images.githubusercontent.com/10763939/145158488-a8d5c02c-4a4e-48ef-b5ad-ad2b3e1671dc.png">
![entities_distribution](https://user-images.githubusercontent.com/10763939/145158501-6ce035fe-516d-465e-b20e-542bba8c851e.png)
    
    
    ![entities_weights](https://user-images.githubusercontent.com/10763939/145158513-2e5b718c-dbd9-4a2e-a6c5-1596116d0413.png)

    ![hdbscan_output](https://user-images.githubusercontent.com/10763939/145158518-633923be-9142-4df2-a86d-4b82f34f03b0.png)

    ![keyword_freq](https://user-images.githubusercontent.com/10763939/145158523-c79cec31-f0bc-4211-b647-8f6b827edac1.png)

    ![keyword_weight_hist](https://user-images.githubusercontent.com/10763939/145158532-7427a984-30ef-41e2-9712-82366c38eabd.png)

    <img width="1434" alt="main_page_screenshot" src="https://user-images.githubusercontent.com/10763939/145158533-4ae2f4f9-1990-43f3-8751-d6c93636cc72.png">

    
    ![hdbscan_output](https://user-images.githubusercontent.com/10763939/145158542-2ca0e9d3-fcfe-4ddf-bad9-53a06c4917f1.png)

    ![keyword_freq](https://user-images.githubusercontent.com/10763939/145158544-b5f69ee5-23cd-40a3-9add-02736c320f38.png)

    ![keyword_weight_hist](https://user-images.githubusercontent.com/10763939/145158550-137d2700-9127-412b-a596-cd79032c9b87.png)

    main_page_screenshot
    <img width="1434" alt="main_page_screenshot" src="https://user-images.githubusercontent.com/10763939/145158661-9f75f493-5ab6-4df7-b504-0443e50e0df6.png">

    ![selenium_crawling](https://user-images.githubusercontent.com/10763939/145158690-70254a57-a9bf-4b3d-880b-321c17fb629c.gif)

    
    ![sent_distribution](https://user-images.githubusercontent.com/10763939/145158705-5feef14f-3a6e-42a9-94ea-e89d72383b74.png)

    ![sent_distribution](https://user-images.githubusercontent.com/10763939/145158710-71de2293-cdff-4e7d-8e3c-5651c0fb8752.png)
    silh_hdbscan
<img width="846" alt="silh_hdbscan" src="https://user-images.githubusercontent.com/10763939/145158721-47ef8084-08ee-4205-88fc-5af4da7df6ca.png">
    
    text_corpus_distribution
    <img width="1045" alt="text_corpus_distribution" src="https://user-images.githubusercontent.com/10763939/145158757-7985bc58-72a1-4dfa-af3f-40f6659e61f8.png">

![text_distr](https://user-images.githubusercontent.com/10763939/145158788-8d025a77-5437-42ad-b5f8-73cf0b27e080.png)

    
    ![text_distr](https://user-images.githubusercontent.com/10763939/145158805-3e0a6e5b-a8d7-4180-a44f-fa408500901c.png)

    ![text_distr](https://user-images.githubusercontent.com/10763939/145158813-ffce1e0d-3c48-4574-9544-f1eec9b1bc19.png)

    ![title_distr](https://user-images.githubusercontent.com/10763939/145158828-04e9b5be-9182-44a2-aa03-1b5e95840ca2.png)
    
    title_distr
<img width="1046" alt="titles_corpus_distribution" src="https://user-images.githubusercontent.com/10763939/145158838-ab5eb337-fe3b-462b-ab03-a34133f4c774.png">


  webcrawler_workflow
    <img width="200" alt="webcrawler_workflow" src="https://user-images.githubusercontent.com/10763939/145158872-8b5ad7b3-617e-4a0f-9ec8-e190da83a466.png">

    
    ![word_freq_corpus](https://user-images.githubusercontent.com/10763939/145158910-352c6830-d0fc-4471-94a7-b6726fe6d4d3.png)

     word_weights_distr
    <img width="1045" alt="word_weights_distr" src="https://user-images.githubusercontent.com/10763939/145158921-3f53d80a-f84c-4648-8edb-a871776d0343.png">
    
    yahoo_cryptos_inspection
    <img width="1193" alt="yahoo_cryptos_inspection" src="https://user-images.githubusercontent.com/10763939/145158943-a3e9e64a-3a47-4ec3-a6db-16b0d5d6de87.png">

    
    yahoo_cryptos
    <img width="1189" alt="yahoo_cryptos" src="https://user-images.githubusercontent.com/10763939/145158969-7c5525ac-1179-4e67-983f-d02bd0a86c92.png">

    
    

    
