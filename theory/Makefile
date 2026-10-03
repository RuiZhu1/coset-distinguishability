.PHONY: pdf check completion search clean
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check:
	python3 checks/exact_small_codes.py
	python3 checks/constitution.py
	python3 checks/handproof_checks.py

completion:
	python3 checks/completion_checks.py
	python3 checks/second_pass_checks.py
	python3 checks/third_pass_checks.py
	python3 checks/circuit_peierls_checks.py

search:
	python3 checks/universal_order_search.py --budget 60 --jobs 4

clean:
	latexmk -C
