"""Reference list for the Review 1 report.

Every entry here was checked in this session by opening the URL given (ACL Anthology, arXiv,
publisher/author page or official documentation) and comparing title, authors and venue.
Entries marked `note` were verified through a secondary record only; see the comment next to them.
Keys are cited in the template as {{key}}; numbers are assigned by order of first citation (IEEE style).
"""

REFS = {
    "sennrich": (
        'R. Sennrich, B. Haddow, and A. Birch, "Neural machine translation of rare words with subword units," '
        "in Proc. 54th Annu. Meeting Assoc. Comput. Linguistics (ACL), Berlin, Germany, Aug. 2016, pp. 1715-1725, "
        "doi: 10.18653/v1/P16-1162. [Online]. Available: https://aclanthology.org/P16-1162/"
    ),
    "kudo": (
        'T. Kudo, "Subword regularization: Improving neural network translation models with multiple subword candidates," '
        "in Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (ACL), Melbourne, Australia, Jul. 2018, pp. 66-75, "
        "doi: 10.18653/v1/P18-1007. [Online]. Available: https://aclanthology.org/P18-1007/"
    ),
    "kudorichardson": (
        'T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer '
        'for neural text processing," in Proc. 2018 Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations '
        "(EMNLP), Brussels, Belgium, Nov. 2018, pp. 66-71, doi: 10.18653/v1/D18-2012. "
        "[Online]. Available: https://aclanthology.org/D18-2012/"
    ),
    "wu": (
        'Y. Wu et al., "Google\'s neural machine translation system: Bridging the gap between human and machine translation," '
        "arXiv:1609.08144, 2016. [Online]. Available: https://arxiv.org/abs/1609.08144"
    ),
    "devlin": (
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for '
        'language understanding," in Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. '
        "Technol. (NAACL-HLT), Minneapolis, MN, USA, Jun. 2019, pp. 4171-4186, doi: 10.18653/v1/N19-1423. "
        "[Online]. Available: https://aclanthology.org/N19-1423/"
    ),
    "song": (
        'X. Song, A. Salcianu, Y. Song, D. Dopson, and D. Zhou, "Fast WordPiece tokenization," in Proc. 2021 Conf. '
        "Empirical Methods Natural Lang. Process. (EMNLP), Online and Punta Cana, Dominican Republic, Nov. 2021, "
        "pp. 2089-2103, doi: 10.18653/v1/2021.emnlp-main.160. [Online]. Available: https://aclanthology.org/2021.emnlp-main.160/"
    ),
    "provilkov": (
        'I. Provilkov, D. Emelianenko, and E. Voita, "BPE-dropout: Simple and effective subword regularization," '
        "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), Online, Jul. 2020, pp. 1882-1892, "
        "doi: 10.18653/v1/2020.acl-main.170. [Online]. Available: https://aclanthology.org/2020.acl-main.170/"
    ),
    "bostrom": (
        'K. Bostrom and G. Durrett, "Byte pair encoding is suboptimal for language model pretraining," in Findings '
        "Assoc. Comput. Linguistics: EMNLP 2020, Online, Nov. 2020, pp. 4617-4624, doi: 10.18653/v1/2020.findings-emnlp.414. "
        "[Online]. Available: https://aclanthology.org/2020.findings-emnlp.414/"
    ),
    "rust": (
        'P. Rust, J. Pfeiffer, I. Vulic, S. Ruder, and I. Gurevych, "How good is your tokenizer? On the monolingual '
        'performance of multilingual language models," in Proc. 59th Annu. Meeting Assoc. Comput. Linguistics and 11th '
        "Int. Joint Conf. Natural Lang. Process. (ACL-IJCNLP), Online, Aug. 2021, pp. 3118-3135, "
        "doi: 10.18653/v1/2021.acl-long.243. [Online]. Available: https://aclanthology.org/2021.acl-long.243/"
    ),
    "mielke": (
        'S. J. Mielke et al., "Between words and characters: A brief history of open-vocabulary modeling and '
        'tokenization in NLP," arXiv:2112.10508, 2021. [Online]. Available: https://arxiv.org/abs/2112.10508'
    ),
    "clark": (
        'J. H. Clark, D. Garrette, I. Turc, and J. Wieting, "Canine: Pre-training an efficient tokenization-free encoder '
        'for language representation," Trans. Assoc. Comput. Linguistics, vol. 10, pp. 73-91, 2022, '
        "doi: 10.1162/tacl_a_00448. [Online]. Available: https://aclanthology.org/2022.tacl-1.5/"
    ),
    "slp3": (
        'D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. (draft), ch. 2, "Words and Tokens." '
        "[Online]. Available: https://web.stanford.edu/~jurafsky/slp3/"
    ),
    "gimpel": (
        'K. Gimpel et al., "Part-of-speech tagging for Twitter: Annotation, features, and experiments," in Proc. 49th '
        "Annu. Meeting Assoc. Comput. Linguistics: Human Lang. Technol. (ACL-HLT), Portland, OR, USA, Jun. 2011, "
        "pp. 42-47. [Online]. Available: https://aclanthology.org/P11-2008/"
    ),
    "owoputi": (
        'O. Owoputi, B. O\'Connor, C. Dyer, K. Gimpel, N. Schneider, and N. A. Smith, "Improved part-of-speech tagging '
        'for online conversational text with word clusters," in Proc. 2013 Conf. North Amer. Chapter Assoc. Comput. '
        "Linguistics: Human Lang. Technol. (NAACL-HLT), Atlanta, GA, USA, Jun. 2013, pp. 380-390. "
        "[Online]. Available: https://aclanthology.org/N13-1039/"
    ),
    "nltk_tweet": (
        'NLTK Project, "nltk.tokenize.casual module (TweetTokenizer)," NLTK documentation. '
        "[Online]. Available: https://www.nltk.org/api/nltk.tokenize.casual.html"
    ),
    "ritter": (
        'A. Ritter, S. Clark, Mausam, and O. Etzioni, "Named entity recognition in tweets: An experimental study," '
        "in Proc. 2011 Conf. Empirical Methods Natural Lang. Process. (EMNLP), Edinburgh, U.K., Jul. 2011, "
        "pp. 1524-1534. [Online]. Available: https://aclanthology.org/D11-1141/"
    ),
    "hanbaldwin": (
        'B. Han and T. Baldwin, "Lexical normalisation of short text messages: Makn sens a #twitter," in Proc. 49th '
        "Annu. Meeting Assoc. Comput. Linguistics: Human Lang. Technol. (ACL-HLT), Portland, OR, USA, Jun. 2011, "
        "pp. 368-378. [Online]. Available: https://aclanthology.org/P11-1038/"
    ),
    "eisenstein": (
        'J. Eisenstein, "What to do about bad language on the internet," in Proc. 2013 Conf. North Amer. Chapter Assoc. '
        "Comput. Linguistics: Human Lang. Technol. (NAACL-HLT), Atlanta, GA, USA, Jun. 2013, pp. 359-369. "
        "[Online]. Available: https://aclanthology.org/N13-1037/"
    ),
    "baldwin": (
        'T. Baldwin, M.-C. de Marneffe, B. Han, Y.-B. Kim, A. Ritter, and W. Xu, "Shared tasks of the 2015 workshop on '
        'noisy user-generated text: Twitter lexical normalization and named entity recognition," in Proc. Workshop '
        "Noisy User-generated Text (WNUT), Beijing, China, Jul. 2015, pp. 126-135, doi: 10.18653/v1/W15-4319. "
        "[Online]. Available: https://aclanthology.org/W15-4319/"
    ),
    "barbieri": (
        'F. Barbieri, J. Camacho-Collados, L. Espinosa Anke, and L. Neves, "TweetEval: Unified benchmark and '
        'comparative evaluation for tweet classification," in Findings Assoc. Comput. Linguistics: EMNLP 2020, '
        "Online, Nov. 2020, pp. 1644-1650, doi: 10.18653/v1/2020.findings-emnlp.148. "
        "[Online]. Available: https://aclanthology.org/2020.findings-emnlp.148/"
    ),
    "nguyen": (
        'D. Q. Nguyen, T. Vu, and A. T. Nguyen, "BERTweet: A pre-trained language model for English tweets," in Proc. '
        "2020 Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations (EMNLP), Online, Oct. 2020, "
        "pp. 9-14, doi: 10.18653/v1/2020.emnlp-demos.2. [Online]. Available: https://aclanthology.org/2020.emnlp-demos.2/"
    ),
    "antypas": (
        'D. Antypas, A. Ushio, J. Camacho-Collados, V. Silva, L. Neves, and F. Barbieri, "Twitter topic classification," '
        "in Proc. 29th Int. Conf. Comput. Linguistics (COLING), Gyeongju, South Korea, Oct. 2022, pp. 3386-3400. "
        "[Online]. Available: https://aclanthology.org/2022.coling-1.299/"
    ),
    "demszky": (
        'D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, "GoEmotions: A dataset of '
        'fine-grained emotions," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), Online, Jul. 2020, '
        "pp. 4040-4054, doi: 10.18653/v1/2020.acl-main.372. [Online]. Available: https://aclanthology.org/2020.acl-main.372/"
    ),
    "zhang": (
        'X. Zhang, J. Zhao, and Y. LeCun, "Character-level convolutional networks for text classification," in Adv. '
        "Neural Inf. Process. Syst. 28 (NIPS 2015), arXiv:1509.01626. [Online]. Available: https://arxiv.org/abs/1509.01626"
    ),
    # Verified through the Semantic Scholar record (title, author, venue, year, DOI); the ACM page returned HTTP 403.
    "thompson": (
        'K. Thompson, "Programming techniques: Regular expression search algorithm," Commun. ACM, 1968, '
        "doi: 10.1145/363347.363387. [Online]. Available: https://doi.org/10.1145/363347.363387"
    ),
    # Verified through the author's abstract page at the University of Wisconsin; the ACM page returned HTTP 403.
    "reps": (
        'T. Reps, "\'Maximal-munch\' tokenization in linear time," ACM Trans. Program. Lang. Syst., 1998. '
        "[Online]. Available: https://research.cs.wisc.edu/wpis/abstracts/toplas98b.abs.html"
    ),
    "mohri": (
        'M. Mohri, "Finite-state transducers in language and speech processing," Comput. Linguistics, vol. 23, no. 2, '
        "pp. 269-311, Jun. 1997. [Online]. Available: https://aclanthology.org/J97-2003/"
    ),
    "ply": (
        'D. Beazley, "PLY (Python Lex-Yacc)," documentation, 2001-2020. '
        "[Online]. Available: https://ply.readthedocs.io/en/latest/ply.html"
    ),
    "pedregosa": (
        'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, '
        "pp. 2825-2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html"
    ),
    "sklearn_tfidf": (
        'scikit-learn developers, "TfidfVectorizer," scikit-learn documentation. '
        "[Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
    ),
    "loper": (
        'E. Loper and S. Bird, "NLTK: The Natural Language Toolkit," in Proc. ACL Workshop Effective Tools Methodologies '
        "Teaching NLP Comput. Linguistics, Philadelphia, PA, USA, Jul. 2002, arXiv:cs/0205028. "
        "[Online]. Available: https://arxiv.org/abs/cs/0205028"
    ),
    "porter": (
        'M. F. Porter, "An algorithm for suffix stripping," Program, vol. 14, no. 3, pp. 130-137, 1980 '
        '(citation as given on the author\'s page, "The Porter stemming algorithm"). '
        "[Online]. Available: https://tartarus.org/martin/PorterStemmer/"
    ),
    "mccallum": (
        'A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," '
        "Just Research and Carnegie Mellon Univ., 1998. "
        "[Online]. Available: https://www.cs.tufts.edu/~nr/cs257/archive/andrew-mccallum/event-models.pdf"
    ),
    "wangmanning": (
        'S. Wang and C. D. Manning, "Baselines and bigrams: Simple, good sentiment and topic classification," in Proc. '
        "50th Annu. Meeting Assoc. Comput. Linguistics (Vol. 2: Short Papers), Jeju Island, South Korea, Jul. 2012, "
        "pp. 90-94. [Online]. Available: https://aclanthology.org/P12-2018/"
    ),
    "lhoest": (
        'Q. Lhoest et al., "Datasets: A community library for natural language processing," arXiv:2109.02846, 2021 '
        "(EMNLP 2021 system demonstrations, as listed on arXiv). [Online]. Available: https://arxiv.org/abs/2109.02846"
    ),
}
