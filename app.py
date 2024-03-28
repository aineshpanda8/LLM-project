# app.py
import streamlit as st
from dotenv import load_dotenv
import openai
import os
import langchain.llms as llms
from langchain.chains import create_sql_query_chain
import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase
import openpyxl
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


# Load environment variables
load_dotenv()

ccc_logo = "CCC logo.png"
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image(ccc_logo, width=100)  # Adjust the width as needed
st.write("---") 
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
st.title("AIOC Guardian")

# User Input Section
st.subheader("Ask a Question:")
question = st.text_area("Enter your question:")

# Initialize database
db = initialize_database()

# Initialize language model
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Initialize SQL query chain
# Updated SQL Prompt
# Define the base prompt
base_prompt = PromptTemplate(
    input_variables=["input"],
    template="""You are a SQLite expert and Machine Learning Engineer. 
    Given an input question, create a syntactically correct SQLite query to run, to answer the input question.
    \n Only use the following tables: {table_info}.Question: {input}.Generate up to {top_k} SQL queries to answer the question.""",
)

# Define the metric prompt
metric_prompt = """**Metric Selection Instructions:**
* For questions about detailed performance metrics (accuracy, MAE, recall, precision), you MUST JOIN the 
'models' table with the 'model_metrics_view' on the MODEL_ID column.
* Check the model type first:
   * For Regression models, use MAE from model_metrics_view.
   * For Classification models, use accuracy, precision, and recall from model_metrics_view.
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
"""

# Define the data validation prompt
data_validation_prompt = """**Data Validation Instructions:**
* Before executing the query, ensure that the referenced data likely exists in the database.
* If potential issues are detected, indicate the problem and suggest alternatives instead of executing the query.
"""


# Prompt to answer the questions
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, format the SQL result in a
      readable way and provide a conversational answer.If the SQL result is empty, provide a helpful message indicating that no matching data was found.  
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
# Combine the prompts using FewShotPromptTemplate
def generate_prompt(query):
    prompt = base_prompt

    # Analyze user query for keywords
    keywords = ["accuracy", "recall", "precision", "MAE", 'model', 'best', 'worst']
    detected_metrics = [metric for metric in keywords if metric in query.lower()]

    # Add conditional prompts based on detected metrics
    if detected_metrics:
        prompt = FewShotPromptTemplate(
            examples=[
                FewShotPromptTemplate.create_example(
                    base_prompt.template,
                    metric_prompt,
                ),
                FewShotPromptTemplate.create_example(
                    base_prompt.template,
                    data_validation_prompt,
                ),
            ],
            prefix=prompt.template,
            suffix="\nSQL Query:",
            example_prompt=base_prompt,
        )

    return prompt



# Initialize Chain 1: Generate SQL Query
generate_sql_chain = create_sql_query_chain(prompt=generate_prompt("dummy_query"), llm=llm, db=db)

execute_query = QuerySQLDataBaseTool(db=db)

# Initialize Chain 2: Execute SQL Query and generate structured answer
execute_sql_chain = answer_prompt | llm | StrOutputParser()

# Main app logic
if st.button("Generate SQL"):
    def calculate_token_size(text):
    # Split text into tokens and count the total number of tokens
        tokens = text.split()
        return len(tokens)
    
    sql_query = generate_sql_chain.invoke({"question": question, "top_k": 1})

    # Step 2: Execute the SQL Query to get the result
    sql_result = execute_query(sql_query)  # Ensure this returns the result of executing the SQL query

    # Step 3: Pass the necessary inputs to the final chain and format the output to include both SQL query and result
    final_response = execute_sql_chain.invoke({"question": question, "query": sql_query, "result": sql_result})

    # Calculate token size of input and output
    input_token_size = calculate_token_size(question)
    output_token_size = calculate_token_size(final_response)
    total_token_size = input_token_size + output_token_size
    
    # Prompt user if total token size exceeds 10K tokens
    if total_token_size > 10000:
        st.warning("Your query is too complex. Please try asking in a simpler way or split it into multiple questions.")
    else:
    # Format the final answer to include both the SQL query and its result
        final_answer = f"SQL Query: {sql_query}\n Answer: {final_response}"
    # Display the results
        st.subheader("Answers:")
        st.write(final_response)
        st.subheader("SQL Query:")
        st.code(sql_query, language="sql")



