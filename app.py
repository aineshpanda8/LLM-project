import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate
from sentence_transformers import SentenceTransformer, util
import base64

# Load environment variables
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3

load_dotenv()

ccc_logo = "CCC_Logo_Light_Blue_(1).jpg"  # Replace with the actual file name of your logo

# Set app configuration
st.set_page_config(layout="wide", page_icon=":guardsman:", page_title="AIOC Guardian")

# Add custom CSS styles
st.markdown(
    """
    <style>
        body {
            background-color: #E6F1F8;
            color: #1c1c1c;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            margin: 0;
        }
        .stApp {
            background-color: #E6F1F8;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            flex: 1;
        }
        .navbar {
            background-color: #0072C6;
            padding: 20px;
            color: #FFFFFF;
        }
        .footer {
            background-color: #0072C6;
            padding: 10px;
            color: #FFFFFF;
            text-align: center;
        }
        .card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
            height: 400px;
            color: #1c1c1c;
        }
        .card:hover {
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
        }
        .card > :first-child {
            text-align: center;
            color: #1c1c1c;
        }
        .stTextArea textarea {
            color: #1c1c1c;
        }
        .stButton button {
            background-color: #FDB913;
            color: #1c1c1c;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #FFC72C;
        }
        h1, h3 {
            color: #0072C6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a navigation bar
st.markdown(
    """
    <div class="navbar">
        <h2>AIOC Guardian</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image(ccc_logo, width=100)  # Adjust the width as needed
    st.write("---")

# User Input Section
col1, col2 = st.columns([1, 1])
col1.markdown("<h3 style='text-align: center'>Ask a Question</h3>", unsafe_allow_html=True)
col2.markdown("<h3 style='text-align: center'>SQL Query</h3>", unsafe_allow_html=True)

# Create two columns for the card layout
col1, col2 = st.columns(2)

# Card 1: Human Language Input
with col1:
    question = st.text_area("Enter your question here...", height=200, key="input_question", label_visibility="collapsed")
    generate_sql_button = st.button("Generate SQL", key="generate_sql_button")

# Card 2: SQL Output
with col2:
    sql_code_placeholder = st.empty()

# Display the final answer
st.markdown("<h3>Answer</h3>", unsafe_allow_html=True)
final_answer_placeholder = st.empty()

# Style the placeholder for scrolling
st.markdown(
    """
    <div style='height: 300px; overflow-y: scroll;'>
        <div id='answer_output'></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add custom CSS styles for the answer output
st.markdown(
    """
    <style>
        #answer_output {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)



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

# Initialize database
db = initialize_database()

# Initialize language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize SQL query chain
base_prompt = PromptTemplate(
    input_variables=["input"],
    template="""You are a SQLite expert and Machine Learning Engineer. Given an input question, create a syntactically correct SQLite query to run, to answer the input question. 
    * Do not use aggregate functions like MAX() on the performance metrics columns, as they are stored as JSON arrays. Instead, directly access the required metric from the 'model_metrics_view'.
\n
**Example:**  

Question: Give company-wise best performing models with metrics.
WITH cte AS (
  SELECT m.Company_Name, m.Model_Name, m.Model_Version, m.Model_Type,
         CASE 
           WHEN m.Model_Type IN ('Multi-Class', 'Classification') THEN mv.accuracy
           WHEN m.Model_Type = 'Regression' THEN mv.MAE
         END AS metric,
         ROW_NUMBER() OVER (PARTITION BY m.Company_Name ORDER BY 
                              CASE 
                                WHEN m.Model_Type IN ('Multi-Class', 'Classification') THEN mv.accuracy
                                WHEN m.Model_Type = 'Regression' THEN -mv.MAE
                              END DESC) AS rn
  FROM models m
  JOIN model_metrics_view mv ON m.MODEL_ID = mv.MODEL_ID
)
SELECT Company_Name, Model_Name, Model_Version, Model_Type, metric
FROM cte
WHERE rn = 1;

\n Only use the following tables: {table_info}.Question: {input}.Generate up to {top_k} SQL queries to answer the question.""",
)

# Define the metric prompt
metric_prompt = """**Metric Selection Instructions:**
* For questions about detailed performance metrics (accuracy, MAE, recall, precision), you MUST JOIN the 'models' table with the 'model_metrics_view' using a CTE (Common Table Expression).
* Use window functions (e.g., ROW_NUMBER(), RANK()) to handle multiple model versions and select the best metrics for each model.
* Handle the performance metrics column correctly by extracting the appropriate metric based on the model type.
* Below are the examples for you to learn on how to write effective code and on how to join the table with model_metrics_view

**Example:**
Question: Which models have the highest recall for each model name?
WITH cte AS (
  SELECT m.Model_Name, m.Model_Version, mv.recall,
         ROW_NUMBER() OVER (PARTITION BY m.Model_Name ORDER BY mv.recall DESC) AS rn
  FROM models m
  JOIN model_metrics_view mv ON m.MODEL_ID = mv.MODEL_ID
  WHERE m.Model_type IN ('Multi-Class', 'Classification')
)

SELECT Model_Name, Model_Version, recall
FROM cte
WHERE rn = 1;

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

Question: What is the volume of Model A in CCC company ?
SQL Query:    SELECT SUM(Daily_Volume) AS total_volume 
         FROM models 
         WHERE lower(Model_Name) = 'model a' and lower(Company_Name) == 'ccc';
    Answer: It seems there is no model named 'Model A' and no company called CCC in the database.  Can you please try with a different model and company name?
"""

# Prompt to answer the questions
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in conversational tone. If the SQL result is empty, provide a helpful message indicating that no matching data was found.
    * Important - Limit the answer to max 5000 tokens
    {question}  <-- Notice the removal of "Question:" 
    SQL Query: {query}
    SQL Result: {result}
    Answer:
    """
)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined reference questions related to performance metrics and model comparison
reference_questions = [
    "What is the accuracy of the model?",
    "Which model has the highest precision?",
    "Compare the recall of different models.",
    "What is the MAE of the regression model?",
    "Which model performs best in terms of accuracy?",
]

# Combine the prompts using FewShotPromptTemplate
def generate_prompt(query):
    prompt = base_prompt 
    # Analyze user query using semantic search
    if should_include_metric_prompt(query):
        prompt = FewShotPromptTemplate(
            examples=[
                {"query": base_prompt.template, "context": metric_prompt},
                {"query": base_prompt.template, "context": data_validation_prompt},
            ],
            example_prompt=base_prompt,
            suffix="\nSQL Query:",
            input_variables=["query"],
        )
    return prompt

def should_include_metric_prompt(query):
    # Encode the user's question and reference questions
    query_embedding = model.encode(query)
    reference_embeddings = model.encode(reference_questions)

    # Compute the cosine similarity between the user's question and reference questions
    similarity_scores = util.cos_sim(query_embedding, reference_embeddings)

    # Check if any similarity score exceeds a threshold (e.g., 0.7)
    if (similarity_scores > 0.7).any():
        return True

    return False

# Main app logic
if generate_sql_button:
    def calculate_token_size(text):
        # Split text into tokens and count the total number of tokens
        tokens = text.split()
        return len(tokens)
    
    # Initialize Chain 1: Generate SQL Query
    generate_sql_chain = create_sql_query_chain(prompt=generate_prompt("dummy_query"), llm=llm, db=db)

    execute_query = QuerySQLDataBaseTool(db=db)

    # Initialize Chain 2: Execute SQL Query and generate structured answer
    execute_sql_chain = answer_prompt | llm | StrOutputParser()
    
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
        final_answer_placeholder.write("Your query is too complex. Please try asking in a simpler way or split it into multiple questions.")
    else:
        # Display the results
        final_answer_placeholder.markdown(f"<div id='answer_output'>{final_response}</div>", unsafe_allow_html=True)
        sql_code_placeholder.code(sql_query, language="sql")

# Add a footer
st.markdown(
    """
    <div class="footer">
        <p>&copy; 2023 CCC Information Services Inc. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
