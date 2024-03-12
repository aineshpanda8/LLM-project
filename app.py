# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.sql_database import SQLDatabase
from operator import itemgetter
import pandas as pd
import sqlite3
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize database
def initialize_database():
    db = SQLDatabase.from_uri("sqlite:///dataset.db")
    return db

# Load CSV file
csv_file = 'dataset_flattened.csv'
df = pd.read_csv(csv_file)

# Save to SQLite DB
db_file = 'dataset.db'
conn = sqlite3.connect(db_file)
table_name = 'models'
df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

# Initialize Streamlit app
st.title("CCCQueryCraft")

# User Input Section
st.subheader("Ask a Question:")
question = st.text_input("Enter your question:")

# Initialize database
db = initialize_database()

# Initialize language model
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Initialize SQL query chain
# Updated SQL Prompt
SQl_prompt_updated = PromptTemplate.from_template(
    """You are a SQLite expert and Machine Learning Engineer. Given an input question, create a syntactically correct SQLite query to run, to answer the input question.

    **Important:** 
    * For questions about detailed performance metrics (accuracy, MAE, recall, precision), **YOU MUST** JOIN the 'models' table with the 'model_metrics_view' (which contains structured performance data) on the MODEL_ID column and **USE ONLY** columns in 'model_metrics_view' to get accuracy, recall, MAE and precsion and also add a WHERE condition based on the model_type .
    * **YOU MUST check if the model type first based on that as a ML engineer analyse which metric is suitable**
       * Regression models: Use MAE from model_metrics_view
       * Classification models: Use accuracy, precision, recall from model_metrics_view.
    * If the question might yield multiple results, ensure your query returns ALL relevant entries.  Only limit output if the user explicitly asks for a 'best' result.
    * Remember, your query should aim to answer the question with a single SQL statement and limit the results appropriately. You can order the results to return the most informative data in the database.
    * **Check Data Availability:** Before providing a final response, ensure that the data referred to in the query (Model_Name, date, company) likely exists in the database. If potential issues are detected, indicate the problem and suggest alternatives for the user instead of executing the query.
    **Before Execution:** Double-check your query for errors. Is the JOIN with 'model_metrics_view' used when appropriate?  Does the query structure seem likely to return the correct number of results? If you identify any potential issues, rewrite the query and rerun the corrected query. 

**Important:** Pay close attention to relationships between tables and views.  JOIN tables using appropriate conditions.  

    **Schema Summary:**
    * models table: MODEL_ID, Model_Name, Model_Type, ... 
    * model_metrics_view: MODEL_ID, recall, precision, accuracy, MAE

    **Example:**  
    Question: Which models have recall greater than 0.9?
    SELECT Model_Name 
         FROM models 
         JOIN model_metrics_view ON models.MODEL_ID = model_metrics_view.MODEL_ID
         WHERE Model_type IN ('Multi-Class', 'Classification') and accuracy > 0.9;
    
    Question: What is the accuracy of Model 10?
     SELECT accuracy 
         FROM models 
         JOIN model_metrics_view ON models.MODEL_ID = model_metrics_view.MODEL_ID
         WHERE lower(Model_Name) = 'model 10' and Model_type IN ('Multi-Class', 'Classification');
    Answer: Model 10 is a Regression model type so it accuracy is not the right metric to calculate performace of the model

    Question: What is the volume of Model A in CCC company ?
    SELECT Daily_Volume 
         FROM models 
         WHERE lower(Model_Name) = 'model a' and lower(Company_Name) == 'ccc';
    Answer: It seems there is no model named 'Model A' and no company called CCC in the database.  Can you please try with a different model and company name?
    
    Only use the following tables: {table_info}.
    Question: {input}.
    Generate up to {top_k} SQL queries to answer the question.
    """
)

# Prompt to answer the questions
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in conversational tone.
      If the SQL result is empty, provide a helpful message indicating that no matching data was found.  

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)



# Initialize Chain 1: Generate SQL Query (this part seems correct based on your setup)
generate_sql_chain = create_sql_query_chain(prompt=SQl_prompt_updated, llm=llm, db=db)

# Initialize SQL execution chain
execute_query = QuerySQLDataBaseTool(db=db)

execute_sql_chain = answer_prompt | llm | StrOutputParser()

# Main app logic
if st.button("Generate SQL"):
    # Step 1: Generate SQL Query
    sql_query = generate_sql_chain.invoke({"question": question, "top_k": 1})
    
    # Step 2: Execute the SQL Query to get the result
    sql_result = execute_query(sql_query)  # Ensure this returns the result of executing the SQL query
    
    # Step 3: Pass the necessary inputs to the final chain and format the output to include both SQL query and result
    final_response = execute_sql_chain.invoke({"question": question, "query": sql_query, "result": sql_result})

    # Display the results
    st.subheader("SQL Query:")
    st.code(sql_query, language="sql")

    st.subheader("Answers:")
    st.write(final_response)

