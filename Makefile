.PHONY: test lint format eval clean

test:
	PYTHONPATH=src:. python3 -m pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/ eval/

format:
	ruff format src/ tests/ scripts/ eval/

eval:
	PYTHONPATH=src:. python3 scripts/run_e2e_eval.py

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf results/ data/index/
	find . -name "*.pyc" -delete
