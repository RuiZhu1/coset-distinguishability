.PHONY: pdf check clean
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check:
	python3 checks/exact_small_codes.py
	python3 checks/constitution.py

clean:
	latexmk -C
