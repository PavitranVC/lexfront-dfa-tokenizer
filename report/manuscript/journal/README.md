# IEEE Access version of the manuscript

`manuscript_access.tex` / `manuscript_access.pdf` is the same paper in the IEEE Access layout. The content comes from `../manuscript.tex`; `build_access.py` regenerates the Access file from it (Access title block, abstract rewritten without abbreviations, section names, drop cap, back matter).

## Build

The IEEE Access LaTeX template is not stored in this repository. Download it from IEEE's page (https://ieeeaccess.ieee.org/authors/preparing-your-article/, "LaTeX template") and copy its files (`ieeeaccess.cls`, `IEEEtran.cls`, `spotcolor.sty`, `logo.png`, `notaglinelogo.png`, `bullet.png` and the `t1-*` font files) into this folder. Then:

    python3 build_access.py
    pdflatex manuscript_access.tex   # run three times

## Before submitting, fill in the placeholders marked `[...]`

- corresponding author and author e-mail addresses, funding information
- repository URL (Data and Code Availability)
- author contributions, competing interests (confirm)
- the three biographies (degree, interests, photograph; the template also accepts `IEEEbiography` with a photo)
- date, volume and DOI lines are IEEE's; the DOI is a placeholder
- confirm the "Use of Generative AI" statement matches the journal's current policy
