# Build the paper with plain pdflatex (no bibtex: the bibliography is in references.tex).
PY ?= $(HOME)/miniconda3/envs/lcd/bin/python
.PHONY: pdf figures clean

pdf: main.pdf

main.pdf: main.tex references.tex figures/fig_surface_rates.pdf figures/fig_free_order.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@! grep -E "undefined|Rerun to get" main.log || (echo "unresolved references"; false)

figures figures/fig_surface_rates.pdf figures/fig_free_order.pdf: figures/make_figures.py ../../results/p1_exchange_rate.json
	$(PY) figures/make_figures.py

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
