# Code from Chapter 01
# Book: Embeddings at Scale

import faiss
import numpy as np
import pyarrow as pa
import vastdb

BUCKET_NAME = "my-bucket"
SCHEMA_NAME = "my-schema"
TABLE_NAME = "my-table"

# Wrong: Single-node architecture
embeddings = np.load("embeddings.npy")  # Doesn't scale
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # In-memory only
index.add(embeddings)


# Right: Distributed-first architecture

session = vastdb.connect(...)

with session.transaction() as tx:
    bucket = tx.bucket(BUCKET_NAME)
    schema = bucket.schema(SCHEMA_NAME) or bucket.create_schema(SCHEMA_NAME)

    # Create the table.
    dimension = 5
    columns = pa.schema(
        [
            ("id", pa.int64()),
            ("vec", pa.list_(pa.field(name="item", type=pa.float32(), nullable=False), dimension)),
            ("vec_timestamp", pa.timestamp("us")),
        ]
    )

    table = schema.table(TABLE_NAME) or schema.create_table(TABLE_NAME, columns)

    # Insert a few rows of data.
    arrow_table = pa.table(schema=columns, data=[...])
    table.insert(arrow_table)

# Scales from millions to trillions with same API
