.PHONY: install run ingest fetch-corpus ui test lint docker-build docker-up clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m scripts.ingest --corpus-dir data/corpus

fetch-corpus:
	python -m scripts.fetch_corpus

ui:
	streamlit run app/ui/streamlit_app.py

test:
	pytest -q

lint:
	ruff check app tests scripts || true

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf data/chroma data/feedback.db data/registry.json .pytest_cache
