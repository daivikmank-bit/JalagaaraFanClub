KIT    ?= .
SECRET ?= practice

run:
	RAKSHA_SECRET=$(SECRET) python main.py --input $(KIT)/dev/input --output out
valid:
	python $(KIT)/validate_output.py --input $(KIT)/dev/input --output out
score:
	python $(KIT)/check_dev.py --output out --show 10
unit:
	pytest -q tests/unit tests/contract
audit:
	python tools/self_audit.py --kit $(KIT)
check: run valid score unit audit

