.PHONY: run test lint evaluate zip
run:
	bash scripts/run_demo.sh
test:
	.venv/bin/python -m pytest -q
lint:
	.venv/bin/ruff check seif tests scripts deploy
evaluate:
	.venv/bin/python scripts/evaluate.py --output docs/evaluation.json
zip:
	.venv/bin/python scripts/package.py
