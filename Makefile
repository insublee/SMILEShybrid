.PHONY: style quality

check_dirs := SMILEShybrid/executors/configs/

style:
	python3 -m black $(check_dirs)
	python3 -m isort $(check_dirs)
	python3 -m flake8 $(check_dirs)
quality:
	python3 -m isort --check-only $(check_dirs)
	python3 -m flake8 $(check_dirs)
