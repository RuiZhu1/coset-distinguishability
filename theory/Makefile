.PHONY: pdf check search clean
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check:
	python3 checks/exact_small_codes.py
	python3 checks/constitution.py

search:
	python3 checks/universal_order_search.py --budget 60 --jobs 4

clean:
	latexmk -C
