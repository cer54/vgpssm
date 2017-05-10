all: vgpt.m vgpt.pdf
vgpt.m: vgpt.nw
	notangle -R$@ vgpt.nw | cpif $@
 vgpt.pdf: vgpt.nw
	noweave -latex -index -delay vgpt.nw | awk -f listfilt > vgpt.tex
	pdflatex vgpt.tex
	pdflatex vgpt.tex
	rm -rf vgpt.aux vgpt.out vgpt.log vgpt.tex vgpt.toc
