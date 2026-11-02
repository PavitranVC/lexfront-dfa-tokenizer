"""Builds manuscript_access.tex (IEEE Access layout) from the conference manuscript's body text (../manuscript.tex).
Requires the IEEE Access LaTeX template files (ieeeaccess.cls etc.) in this folder, see README.md."""
import re
src = open('../manuscript.tex').read()
body = src[src.index('\\section{Introduction}'):src.index('\\clearpage\n\\balance')]
bib = src[src.index('\\begin{thebibliography}'):src.index('\\end{thebibliography}') + len('\\end{thebibliography}')]

def rep(a, b):
    global body
    assert a in body, a[:70]
    body = body.replace(a, b, 1)

# paper wording instead of report wording
rep('The work in this report makes four contributions.', 'This paper makes four contributions.')
body = body.replace('this report', 'this paper')
# section names a journal reader expects
rep('\\section{Literature Survey}', '\\section{Related Work}')
rep('\\section{Proposed System}', '\\section{System Design}')
rep('\\section{Results and Implementation}', '\\section{Implementation and Results}')
rep('\\section{Performance and Metrics}', '\\section{Comparison of the Existing and Proposed Setups}')
# drop cap on the first word of the introduction
m = re.match(r'(\\section\{Introduction\}\\label\{sec:1\}\n)(\w)(\w*)', body)
body = body[:m.start()] + m.group(1) + '\\PARstart{%s}{%s}' % (m.group(2), m.group(3)) + body[m.end():]
# existing / proposed defined in the introduction, in the section 3 methodology text and in the abstract
rep('while keeping everything else in the pipeline fixed.', 'while keeping everything else in the pipeline fixed. Throughout the paper the existing setup is the NLTK tokenizer with Naive Bayes, and the proposed setup is LexFront with Logistic Regression.')

abstract = ("Text from social media is full of hashtags, mentions, links, emoticons and stretched-out words, and most tokenizers were not designed for it. "
 "We ask whether a lexical analyzer built the way a compiler builds one can do better. We built LexFront, a lexer with eleven token classes, longest-match scanning, error recovery, a symbol table and six switchable normalization rules, together with our own engine that turns regular expressions into a finite automaton and minimizes it, checked against the lexer built with Python Lex-Yacc that the experiments use. "
 "We call the Natural Language Toolkit tokenizer with Naive Bayes the existing setup and LexFront with Logistic Regression the proposed setup. Tokenizers are compared with the classifier held fixed, on three core datasets, a collection of eight cleaned datasets and four large datasets. "
 "The proposed setup is ahead of the existing one by 17.58 accuracy points on one core dataset and by 1.50 on another, and 0.26 behind on the third, but almost all of that comes from changing the classifier. "
 "With the tokenizer as the only change, LexFront stays within 0.00 to +0.46 accuracy points of the Natural Language Toolkit tokenizer on the large datasets and is clearly ahead only of whitespace splitting. "
 "It gives the lowest out-of-vocabulary rate and the smallest vocabulary among the word-level tokenizers. Subword tokenizers and a character n-gram baseline did not help. We report confidence intervals and the hypotheses that did not hold.")
assert 150 <= len(abstract.split()) <= 250, len(abstract.split())
print('abstract words', len(abstract.split()))

aff = 'School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore, India'
head = r'''\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[section]{placeins}
\usepackage{url}
\setcounter{dbltopnumber}{4}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\topfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\dblfloatpagefraction}{0.7}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\algnewcommand\algorithmicinput{\textbf{input:}}
\hyphenation{Lex-Front tokenizers}
\graphicspath{{../../figures/vector/}}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}
\vol{14}\year{2026}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2026.XXXXXXX}
\title{LexFront: A Compiler-Style Lexer Used as a Pre-Tokenizer for Noisy English Text Classification}
\author{\uppercase{Veer Jahaan Arora}\authorrefmark{1}, \uppercase{Yash Aggarwal}\authorrefmark{1}, AND \uppercase{V C Pavitran}\authorrefmark{1}}
\address[1]{%(aff)s (e-mail: [to be added])}
\tfootnote{This work was carried out for the Compiler Design Lab (BCSE307P). Faculty: Suganthini C. [Funding information, if any, to be added.]}
\markboth{Arora, Aggarwal and Pavitran: LexFront: A Compiler-Style Lexer as a Pre-Tokenizer for Noisy Text Classification}{Arora, Aggarwal and Pavitran: LexFront: A Compiler-Style Lexer as a Pre-Tokenizer for Noisy Text Classification}
\corresp{Corresponding author: V C Pavitran (e-mail: [to be added]).}
\begin{abstract}
%(abstract)s
\end{abstract}
\begin{keywords}
Finite automata, lexical analysis, noisy text, subword tokenization, text classification, tokenization
\end{keywords}
\titlepgskip=-21pt
\maketitle
''' % {'aff': aff, 'abstract': abstract}

tail = r'''
\section*{Data and Code Availability}
The code, configuration files and saved result files behind every table and figure are available in the project repository [repository URL to be added]. The datasets used are public, and each dataset is cited in the text.

\section*{Author Contributions}
[To be completed by the authors, for example in the CRediT format.]

\section*{Competing Interests}
The authors declare no competing interests. [To be confirmed by the authors.]

\section*{Use of Generative AI}
An AI assistant (Anthropic's Claude) was used during the project to help write and review code, to draft and edit parts of the text, and to produce figures. All reported numbers come from running the project's code, and the authors reviewed the text and take responsibility for the content.

\section*{Acknowledgment}
The authors thank their faculty, Suganthini C, for guidance during the Compiler Design Lab.

''' + bib + r'''

\begin{IEEEbiographynophoto}{Veer Jahaan Arora}
is a student in the School of Computer Science and Engineering at VIT, Vellore, India (registration number 24BCB0080). [Degree, research interests and photograph to be added.]
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Yash Aggarwal}
is a student in the School of Computer Science and Engineering at VIT, Vellore, India (registration number 24BKT0013). [Degree, research interests and photograph to be added.]
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{V C Pavitran}
is a student in the School of Computer Science and Engineering at VIT, Vellore, India (registration number 24BCE2019). [Degree, research interests and photograph to be added.]
\end{IEEEbiographynophoto}

\EOD
\end{document}
'''
open('manuscript_access.tex', 'w').write(head + body + tail)
