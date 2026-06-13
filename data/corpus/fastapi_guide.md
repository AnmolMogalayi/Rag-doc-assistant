# FastAPI Guide

FastAPI is a modern, high-performance Python web framework for building APIs, based on standard
Python type hints. It uses Starlette for the web layer and Pydantic for data validation, and it
generates interactive OpenAPI documentation automatically.

## Defining Endpoints

You declare path operations with decorators on an `APIRouter` or the `FastAPI` app:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Path parameters (`item_id`) and query parameters (`q`) are parsed and validated from the type
hints. A request body is declared by typing a parameter as a Pydantic model.

## Request Validation

When a parameter is a Pydantic model, FastAPI validates the JSON body against it. If validation
fails, FastAPI automatically returns **HTTP 422 Unprocessable Entity** with a structured error
describing which fields were invalid — you do not write that code yourself.

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
def query(req: QueryRequest):
    return {"q": req.question}
```

## HTTP Status Codes and Errors

Set a default success code with the `status_code` argument on the decorator, and raise
`HTTPException` for error conditions:

```python
from fastapi import HTTPException, status

@app.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest():
    if nothing_to_do:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Provide a file or URL.")
```

Common codes: `200 OK`, `201 Created`, `400 Bad Request`, `404 Not Found`,
`413 Request Entity Too Large`, `422 Unprocessable Entity`, `500 Internal Server Error`,
`503 Service Unavailable`.

## File Uploads

To accept uploaded files, use `UploadFile` with `File`, and install `python-multipart`:

```python
from fastapi import UploadFile, File

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    for f in files:
        content = await f.read()
```

## Dependency Injection

FastAPI's dependency-injection system lets you declare reusable dependencies with `Depends`:

```python
from fastapi import Depends

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
def users(db = Depends(get_db)):
    ...
```

Dependencies are resolved per-request and can themselves depend on other dependencies
(sub-dependencies).

## Middleware and Lifespan

Use `@app.middleware("http")` to wrap every request (for example, to attach a correlation/trace
id and log latency). Use the **lifespan** context manager to run startup and shutdown logic, such
as warming a model or initializing a database:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown

app = FastAPI(lifespan=lifespan)
```

## Running the Application

Run with an ASGI server such as Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs are served at `/docs` (Swagger UI) and `/redoc` (ReDoc).
