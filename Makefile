.PHONY: build clean default fmt tests

default: fmt | tests

fmt:
	find . -type f -iname "*.py" -print0 | xargs -0 autoflake --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables --in-place 
	black shaprank/ examples/ tests/
	isort shaprank/ examples/*py tests/
	mypy shaprank/
	flake8 shaprank/
	pylint --fail-under=7 shaprank/


build: |
	python -m build --wheel

clean: |
	-rm -rf ./build ./.coverage ./dist ./shaprank.egg-info ./.mypy_cache ./.pytest_cache
	-find . -type d -name '__pycache__' -print0 | xargs -0 rm -rf
	-find . -type d -name '.ipynb_checkpoints' -print0 | xargs -0 rm -rf
	jupyter nbconvert --clear-output --inplace examples/*.ipynb

tests: |
	coverage run -m pytest ./tests
	coverage report -m