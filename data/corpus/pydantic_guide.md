# Pydantic Guide

Pydantic is a data-validation library that uses Python type annotations to validate, parse, and
serialize data. Pydantic v2 is built on a fast Rust core (`pydantic-core`).

## Models

A model is a class that inherits from `BaseModel`. Fields are declared as annotated attributes:

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = "anonymous"
    is_active: bool = True
```

Creating `User(id="123")` coerces the string to an int. Invalid data raises a
`ValidationError` describing every problem at once.

## Fields and Constraints

Use `Field` to add metadata and constraints such as bounds, lengths, and descriptions:

```python
from pydantic import BaseModel, Field

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
```

`ge`/`le` set numeric bounds; `min_length`/`max_length` constrain strings and collections.
Validation errors here surface as HTTP 422 when used with FastAPI.

## Validators

Custom validation uses `field_validator` (single field) and `model_validator` (whole model):

```python
from pydantic import BaseModel, field_validator

class Query(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()
```

## Settings Management

`pydantic-settings` provides `BaseSettings` for 12-factor configuration loaded from environment
variables and `.env` files:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    llm_provider: str = "openai"
    top_k: int = 5
```

Environment variables override defaults; names are matched case-insensitively. This keeps
secrets out of code.

## Serialization

Convert a model to a dict or JSON with `model_dump()` and `model_dump_json()`. Parse incoming
data with `Model(**data)` or `Model.model_validate(data)`. In Pydantic v2 the older `.dict()`
and `.json()` methods are deprecated in favor of `model_dump()` / `model_dump_json()`.

## Structured LLM Output

Because Pydantic models describe a strict schema, they pair well with LLM "structured output":
frameworks like LangChain expose `llm.with_structured_output(MyModel)` to force the model to
return data matching the model, which is ideal for graders that must return a binary score.
