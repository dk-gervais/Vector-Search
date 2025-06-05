import os, pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from dotenv import load_dotenv
load_dotenv(override=True)

username = '_SYSTEM'
password = 'SYS'
hostname = 'iris'
port = 1972
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

engine = create_engine(CONNECTION_STRING)

# Load JSONL file into DataFrame
file_path = './data/financial/tweets_all.jsonl'
df_tweets = pd.read_json(file_path, lines=True)
pd.set_option('display.max_rows', 1000)

with engine.connect() as conn:
    with conn.begin():# Load 
        sql = f"""
                CREATE TABLE financial_tweets_sql (
        note VARCHAR(255),
        sentiment INTEGER,
        note_vector VECTOR(DOUBLE, 384)
        )
                """
        result = conn.execute(text(sql))

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all tweets at once. Batch processing makes it faster
embeddings = model.encode(df_tweets['note'].tolist(), normalize_embeddings=True)

# Add the embeddings to the DataFrame
df_tweets['note_vector'] = embeddings.tolist()

with engine.connect() as conn:
    with conn.begin():
        for index, row in df_tweets.iterrows():
            sql = text("""
                INSERT INTO financial_tweets_sql 
                (note, sentiment, note_vector) 
                VALUES (:note, :sentiment, TO_VECTOR(:note_vector))
            """)
            conn.execute(sql, {
                'note': row['note'], 
                'sentiment': row['sentiment'],
                'note_vector': str(row['note_vector'])
            })

note_search = "Beyond Meat"
search_vector = model.encode(note_search, normalize_embeddings=True).tolist() # Convert search phrase into a vector

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 3 * FROM financial_tweets_sql
            ORDER BY VECTOR_DOT_PRODUCT(note_vector, TO_VECTOR(:search_vector)) DESC
        """)

        results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()

print(results)

results_df = pd.DataFrame(results, columns=df_tweets.columns).iloc[:, :-1] # Remove vector
pd.set_option('display.max_colwidth', None)  # Easier to read description
results_df.head()

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 3 * FROM financial_tweets_sql
            WHERE sentiment = 1
            ORDER BY VECTOR_DOT_PRODUCT(note_vector, TO_VECTOR(:search_vector, double)) DESC
        """)

        results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()
        
print(results)

results_df = pd.DataFrame(results, columns=df_tweets.columns).iloc[:, :-1] # Remove vector
pd.set_option('display.max_colwidth', None)  # Easier to read description
results_df.head()