-- RUN 1st: create vector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- RUN 2nd: create mp table
CREATE TABLE IF NOT EXISTS mp (
  id BIGSERIAL PRIMARY KEY,
  content_title TEXT,
  content_url TEXT,
  content_date TEXT,
  content TEXT,
  content_length BIGINT,
  content_tokens BIGINT,
  embedding VECTOR(1536)
);

-- RUN 3rd: create mp_search function
CREATE OR REPLACE FUNCTION mp_search (
  query_embedding VECTOR(1536),
  similarity_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  content_title TEXT,
  content_url TEXT,
  content_date TEXT,
  content TEXT,
  content_length BIGINT,
  content_tokens BIGINT,
  similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY SELECT
    mp.id,
    mp.content_title,
    mp.content_url,
    mp.content_date,
    mp.content,
    mp.content_length,
    mp.content_tokens,
    1 - (mp.embedding <=> query_embedding) AS similarity
  FROM mp
  WHERE 1 - (mp.embedding <=> query_embedding) > similarity_threshold
  ORDER BY mp.embedding <=> query_embedding DESC
  LIMIT match_count;
END;
$$;

-- RUN 4th: create index for mp table
CREATE INDEX IF NOT EXISTS mp_embedding_index
ON mp
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
