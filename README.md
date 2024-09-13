# LLM-Social-Media-Mental-Analysis

This repository contains the code for the project *Constructing a Mental Health Analysis System for Social Media Using Large Language Models*.

## Abstract
As social media platforms have penetrated into every aspect of people's lives, many mental health problems have also arisen along with them. In today's digital age, analyzing mental health trends through these platforms has become critical. In this study, we present a system designed to identify the mental health trends of Weibo users by extracting and analyzing the content posted by users on Weibo, China's leading social media platform. The system is mainly composed of two parts: data acquisition module and analysis module. The data collection module uses the Python-based web scraping tool Scrapy to scrape comments from popular topics on Weibo. At the heart of the analysis module is a large language model fine-tuned from a psychological database. The module assesses the topic and specific content of the posts, scoring comments based on criteria such as positivity, alignment with mood disorders, and potential signs of psychoactive substance use. This data is stored and mediated using the relational database MySQL, and then analyzed and visualized using advanced data analysis tools. Through this method, we can timely and comprehensively monitor the mental health status of social media platforms, and provide a solid foundation for further academic research on public mental health.


## Environment

- **Operating System**: Ubuntu 24.04  
- **Python**: 3.10.2  
- **MySQL**: 8.0.39-0ubuntu0.24.04.1  

### LLM Model Requirements:

- **GPU Memory**: > 8GB  
- **PyTorch**: 2.3.1  
- **Transformers**: 4.42.4  
- **mysql_connector_repackaged**: 0.3.1  

### Scraping Model Requirements:

- **pymongo**: 4.8.0  
- **PyMySQL**: 1.1.1  
- **Requests**: 2.32.3  
- **Scrapy**: 2.11.2  
- **schedule**: 1.2.2  

## Running the Models

Both models can run on two different machines or a single machine.

### Scraping Model:

1. Create a database using MySQL.
2. Update the MySQL settings in `Weibo_Spider/weibo/setting.py`.
3. Obtain a cookie for `weibo.cn` to enable scraping.
4. Run `default_schedule.py` to start the Scrapy scheduler.

### LLM Model:

1. Update the MySQL settings in `weibo_main.py`.
2. Specify the model path in `glm.py`.
3. Note: Due to the large size (over 20GB) of the pretrained model, it cannot be uploaded with the code. The default setting will automatically download the original GLM4-7b-chat model from Hugging Face.
4. Run `weibo_main.py` to start the LLM model. The results will be stored in the MySQL database.
