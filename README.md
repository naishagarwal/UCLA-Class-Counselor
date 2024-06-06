CS 263 Final Project - Creating Persona Chatbots  
Contributors:  
Naisha Agarwal  
Tristan Herink  
Matthew Yang  
Minrui Gui  

Follow the steps below to spin up a local instance of an Elasticsearch database, embed the class data (embedding_input_class_data.txt), and embed an individual query to return similar results:  

1. Download the lastest version of Elasticsearch from https://www.elastic.co/downloads/elasticsearch  
2. Once it's downloaded, you can run bin/elasticsearch in your terminal to spin up the database  
3. Look in the terminal logs and copy the password then sub it in on line 12 in create_es_index_and_embed.py  
4. Run create_es_index_and_embed.py (this should take ~20 minutes)  
5. Run query_embeddings.py and change out line 45 for whichever query you want to embed and return similar results for
