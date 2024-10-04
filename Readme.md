Steps to Run the project -

1. Clone the project and open project in VS code

2. Create virtual environment using following command - "python -m venv .venv"

3. Press ctrl+shift+p and select python interpreter , by going into scripts folder of virtual environment and selecting python interpreter

4. After this run this command to install requirements - " pip install -r requirements.txt "

5. Create folder in root directory named "pdf-docs" and add pdf of materail on which chatbot is needed to answer questions

6. Run "4.demo_indexing_using_vectorDB.py" file after completion of step 5 , and verify folder named "faiss-index" is created after completion of step 6

7. After this run following command " streamlit run f1.py "

8. Now the chatbot is running on streamlit web interface and questions can be asked to the chatbot

+ Note : Chatbot will provide answers to questions belonging to material on which chatbot is trained and reference to generate answer is also provided by chatbot at end of answer

