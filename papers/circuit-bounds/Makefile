PY ?= $(HOME)/miniconda3/envs/lcd/bin/python

.PHONY: pdf figures clean

pdf: main.pdf

main.pdf: main.tex figures/bounds_vs_distance.pdf figures/hardware_margins.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@! grep -E "undefined|Rerun to get" main.log

# figures are committed alongside; regenerate from the results JSON with `make figures`
figures:
	$(PY) figures/make_figures.py

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
