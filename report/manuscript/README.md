# IEEE manuscript

`manuscript.tex` is the two-column IEEE conference-format paper (IEEEtran). Figures are the vector PDFs in `../figures/vector/`.

Build (TeX Live with the `IEEEtran`, `algorithms`, `algorithmicx`, `booktabs`, `tabularx` and `adjustbox` packages):

    pdflatex manuscript.tex   # run three times so cross references settle

`manuscript.pdf` is the compiled result. The text and numbers come from the final report (`../LexFront_Final_Report.docx`).
