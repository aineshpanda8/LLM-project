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
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Initialize SQL query chain
SQl_prompt = PromptTemplate.from_template(
    """You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run, to answer the input question.
The "Performance Metrics" column in the database stores a list of three values for each entry: [accuracy, precision, MAE]. To query based on one of these metrics, you will need to use the below query.
Select json_extract("Performance Metrics", "$[0]") AS accuracy, json_extract("Performance Metrics", "$[1]") as precision, json_extract("Performance Metrics", "$[2]") AS MAE
For questions about accuracy, focus on the first value in the list. For precision, the second, and for MAE, the third.
Remember, your query should aim to answer the question with a single SQL statement and limit the results appropriately.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
Never query for all columns from a table. Only query the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Be careful not to query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use the date('now') function to get the current date if the question involves "today".
Only use the following tables:

{table_info}.

Question: {input}
Provide up to {top_k} SQL variations."""
)


# Initialize Chain 1: Generate SQL Query (this part seems correct based on your setup)
generate_sql_chain = create_sql_query_chain(prompt=SQl_prompt, llm=llm, db=db)

# Initialize SQL execution chain
execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
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

