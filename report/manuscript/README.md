# IEEE manuscript

`manuscript.tex` is the two-column IEEE conference-format paper (IEEEtran). Figures are the vector PDFs in `../figures/vector/`.

Build (TeX Live with the `IEEEtran`, `algorithms`, `algorithmicx`, `booktabs`, `tabularx` and `adjustbox` packages):

    pdflatex manuscript.tex   # run three times so cross references settle

`manuscript.pdf` is the compiled result.

`manuscript_single.tex` / `manuscript_single.pdf` is the same paper in IEEEtran's single-column journal layout (12 pt). Build it the same way:

    pdflatex manuscript_single.tex

For the double-spaced submission-draft look, change the class options to `[journal,draftcls,onecolumn,12pt,a4paper]` (about 27 pages).
 The text and numbers come from the final report (`../LexFront_Final_Report.docx`).
