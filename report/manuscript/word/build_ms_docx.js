// Builds the IEEE-style Word versions of the manuscript from content.json (blocks of the final report) and manuscript.aux (numbering).
// usage (from report/manuscript/word): node build_ms_docx.js two|single out.docx   (needs `npm install docx`; figures come from ../../figures)
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, WidthType,
  BorderStyle, SectionType, Tab, LevelFormat, ShadingType, VerticalAlign,
} = require('docx');

const MODE = process.argv[2];                    // 'two' or 'single'
const OUT = process.argv[3];
const TWO = MODE === 'two';
const FIGDIR = require('path').join(__dirname, '..', '..', 'figures') + '/';
const FONT = 'Times New Roman';
const D = JSON.parse(fs.readFileSync(__dirname + '/content.json', 'utf8'));
const aux = fs.readFileSync(__dirname + '/manuscript.aux', 'utf8');

// ---- numbering from the LaTeX run, so both versions agree ----
const LAB = {};
for (const m of aux.matchAll(/\\newlabel\{(tab|fig|alg|sec):([^}]+)\}\{\{(.*?)\}\{/g)) {
  LAB[m[1] + ':' + m[2]] = m[3].replace(/\\mbox\s*\{/g, '').replace(/[{}]/g, '');
}
const lab = (kind, id) => LAB[kind + ':' + id] || id;

// ---- sizes (half-points) and widths (twips) ----
const S = TWO ? { body: 20, cap: 16, tab: 15, ref: 16, h: 20, algo: 16, abs: 18, title: 48, auth: 22 }
              : { body: 22, cap: 18, tab: 18, ref: 18, h: 22, algo: 18, abs: 20, title: 48, auth: 24 };
const PAGE_W = 11906, PAGE_H = 16838;
const MARG = TWO ? { top: 1080, bottom: 1440, left: 1008, right: 1008 } : { top: 1440, bottom: 1440, left: 1260, right: 1260 };
const TEXT_W = PAGE_W - MARG.left - MARG.right;
const GAP = 360;
const COL_W = TWO ? Math.floor((TEXT_W - GAP) / 2) : TEXT_W;
const px = (twips) => Math.round(twips / 15);      // twips -> docx image pixels (96 dpi)

// ---- text helpers ----
function fixText(s) {
  s = s.replace(/Sections ([\d.]+(?:(?:, | to | and )[\d.]+)*)/g, (m, list) => 'Sections ' + list.replace(/\d+(?:\.\d+)?/g, (n) => lab('sec', n)));
  s = s.replace(/Section (\d(?:\.\d)?)/g, (m, n) => 'Section ' + lab('sec', n));
  s = s.replace(/Table (\d+)/g, (m, n) => 'Table ' + lab('tab', n));
  s = s.replace(/Figure (\d+)/g, (m, n) => 'Fig. ' + lab('fig', n));
  s = s.replace(/Algorithms (\d) and (\d)/g, (m, a, b) => `Algorithms ${lab('alg', a)} and ${lab('alg', b)}`);
  s = s.replace(/Algorithm (\d)/g, (m, n) => 'Algorithm ' + lab('alg', n));
  s = s.replace(/\+-/g, '±');
  s = s.replace(/(?<![\w\])])-(?=\d)/g, '−');
  s = s.replace(/"([^"]+)"/g, '“$1”');
  return s;
}
function runsOf(rs, size = S.body, extra = {}) {
  return rs.map((r) => new TextRun({
    text: fixText(r.t), font: r.code ? 'Courier New' : FONT, size: r.code ? size - 2 : size,
    bold: !!r.b || extra.bold, italics: !!r.i || extra.italics, ...extra.keep,
  }));
}
const para = (rs) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: TWO ? 228 : 276 }, indent: { firstLine: 245 },
  children: runsOf(rs),
});
const sectionHead = (text) => {
  const m = text.match(/^(\d+)\. (.*)$/);
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `${lab('sec', m[1])}. ${m[2]}`.toUpperCase().replace('F1', 'F1'), font: FONT, size: S.h, smallCaps: false })],
  });
};
const subHead = (text) => {
  const m = text.match(/^(\d+\.\d+) (.*)$/);
  const letter = lab('sec', m[1]).split('-').pop();
  return new Paragraph({
    keepNext: true, spacing: { before: 100, after: 40 },
    children: [new TextRun({ text: `${letter}. ${fixText(m[2])}`, font: FONT, size: S.h, italics: true })],
  });
};

// ---- captions, tables, figures ----
function stripCaption(c) {
  c = c.replace(/^(Table|Figure) \d+\.\s*/, '');
  c = c.replace(/\s*\((?:[^()]*?)(?:results|report)\/[^()]*\)/g, '').replace(/\s*\(redrawn[^)]*\)/g, '');
  return fixText(c.trim());
}
const capNum = (c) => parseInt(c.match(/^(?:Table|Figure) (\d+)\./)[1], 10);
const tabCaption = (n, text) => [
  new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
    children: [new TextRun({ text: `TABLE ${lab('tab', n)}`, font: FONT, size: S.cap })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
    children: [new TextRun({ text: text.toUpperCase(), font: FONT, size: S.cap - 1 })] }),
];
const figCaption = (n, text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 40, after: 120 },
  children: [new TextRun({ text: `Fig. ${lab('fig', n)}. ${text}`, font: FONT, size: S.cap })],
});
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function tableBlock(rows, widthTw) {
  const n = rows[0].length;
  const len = Array.from({ length: n }, (_, j) => Math.max(...rows.map((r) => String(r[j]).length), 4));
  const wts = len.map((l) => Math.min(Math.max(l, 6), 38));
  const tot = wts.reduce((a, b) => a + b, 0);
  let cols = wts.map((w) => Math.floor(widthTw * w / tot));
  cols[cols.length - 1] += widthTw - cols.reduce((a, b) => a + b, 0);
  const thick = { style: BorderStyle.SINGLE, size: 8, color: '000000' }, thin = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
  return new Table({
    width: { size: widthTw, type: WidthType.DXA }, columnWidths: cols,
    rows: rows.map((r, i) => new TableRow({
      cantSplit: true, tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: cols[j], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        borders: { top: i === 0 ? thick : none, bottom: i === 0 ? thin : (i === rows.length - 1 ? thick : none), left: none, right: none },
        margins: { top: 20, bottom: 20, left: 50, right: 50 },
        children: [new Paragraph({ keepNext: i < rows.length - 1 && rows.length <= 14, children: [new TextRun({ text: fixText(String(c)), font: FONT, size: S.tab, bold: i === 0 })] })],
      })),
    })),
  });
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function imageBlock(name, widthTw) {
  const file = FIGDIR + name;
  const [w, h] = pngSize(file);
  const wpx = px(widthTw);
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 20 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(file), transformation: { width: wpx, height: Math.round(wpx * h / w) },
      altText: { title: name, description: name, name } })],
  });
}

// ---- IEEE-style algorithm blocks ----
const KW = new Set(['function', 'while', 'for', 'if', 'else', 'return', 'repeat', 'until', 'in', 'input', 'break', 'continue', 'not']);
function algoRuns(text, sz) {
  text = text.replace(/<-/g, '←').replace(/->/g, '→').replace(/#\s?/, '▷ ');
  const idx = text.indexOf('▷');
  const code = idx >= 0 ? text.slice(0, idx) : text, note = idx >= 0 ? text.slice(idx) : '';
  const runs = []; let first = true;
  code.split(/(\b[A-Za-z_]+\b)/).forEach((tok) => {
    if (tok === '') return;
    const kw = KW.has(tok) || (tok === 'end' && first);
    if (tok.trim() !== '') first = false;
    runs.push(new TextRun({ text: tok, font: FONT, size: sz, bold: kw }));
  });
  if (note) runs.push(new TextRun({ text: '   ' + note, font: FONT, size: sz - 2, italics: true }));
  return runs;
}
function algoParas(num, title, lines, sz) {
  while (lines.length && lines[lines.length - 1] === '') lines = lines.slice(0, -1);
  const thick = { style: BorderStyle.SINGLE, size: 12, color: '000000', space: 3 };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: '000000', space: 2 };
  const out = [new Paragraph({ spacing: { before: 60, after: 30 }, keepNext: true, border: { top: thick, bottom: thin },
    children: [new TextRun({ text: `Algorithm ${num}`, font: FONT, size: sz + 1, bold: true }), new TextRun({ text: ` ${title}`, font: FONT, size: sz + 1 })] })];
  let n = 0;
  lines.forEach((l, i) => {
    const last = i === lines.length - 1; const border = last ? { bottom: thick } : undefined;
    if (l.trim() === '') { out.push(new Paragraph({ spacing: { after: 0, line: 100 }, children: [] })); return; }
    const lead = l.length - l.trimStart().length; const level = Math.min(Math.floor(lead / 4), 6);
    const left = 330 + level * 230;
    if (l.trim().startsWith('#')) {
      out.push(new Paragraph({ spacing: { after: 0, line: 220 }, indent: { left }, border, keepNext: !last, keepLines: true,
        children: [new TextRun({ text: '▷ ' + l.trim().replace(/^#\s?/, ''), font: FONT, size: sz - 2, italics: true })] }));
      return;
    }
    n += 1;
    out.push(new Paragraph({ spacing: { after: 0, line: 220 }, indent: { left, hanging: left }, border, keepNext: !last, keepLines: true,
      children: [new TextRun({ text: `${n}:`, font: FONT, size: sz - 2 }), new TextRun({ children: [new Tab()] }), ...algoRuns(l.trim(), sz)] }));
  });
  return out;
}
function algoPair(a, b) {
  const half = Math.floor((TEXT_W - 200) / 2);
  const cell = (children, right) => new TableCell({ width: { size: half, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none },
    margins: { top: 0, bottom: 0, left: right ? 100 : 0, right: right ? 0 : 100 }, children });
  return [new Table({ width: { size: half * 2, type: WidthType.DXA }, columnWidths: [half, half],
    rows: [new TableRow({ cantSplit: true, children: [cell(algoParas(a.num, a.title, a.lines, S.algo - 1), false), cell(algoParas(b.num, b.title, b.lines, S.algo - 1), true)] })] }),
    new Paragraph({ spacing: { after: 80 }, children: [] })];
}

// ---- walk the blocks into sections ----
const MAIN_FIGS = new Set([1, 3, 4, 5, 6, 7]);
const MAIN_TABS = new Set([1, 3, 10, 12, 13, 14, 15]);
const WIDE_FIG = new Set(['architecture.png', 'architecture_nomod.png', 'tokenizers_accuracy.png', 'hq_module1_accuracy.png', 'hq_existing_vs_proposed.png', 'hq_intrinsic_metrics.png', 'ablation.png', 'subword.png']);
const FIGFILE = (f) => f;                                  // PNG names in content.json are already the PNG files

const B = D.blocks;
const body = [];       // {kind:'flow'|'wide', items:[...]}
const appendix = [];
let algoNo = 0;
let abstractParas = [], keywords = null;

function push(target, kind, items) { target.push({ kind, items }); }

let i = 0, inAbstract = false;
while (i < B.length) {
  const b = B[i];
  if (b.type === 'h1' && b.text === 'Abstract') { inAbstract = true; i++; continue; }
  if (b.type === 'h1' && b.text === 'References') break;
  if (b.type === 'h1') { inAbstract = false; push(body, 'flow', [sectionHead(b.text)]); i++; continue; }
  if (inAbstract) {
    if (b.type === 'p') { if (b.runs[0] && b.runs[0].t.startsWith('Keywords')) keywords = b.runs.slice(1).map((r) => r.t).join('').replace(/\.$/, ''); else abstractParas.push(b); }
    i++; continue;
  }
  if (b.type === 'gap') { i++; continue; }
  if (b.type === 'h2') { push(body, 'flow', [subHead(b.text)]); i++; continue; }
  if (b.type === 'h3') { push(body, 'flow', [new Paragraph({ keepNext: true, spacing: { before: 60, after: 20 }, children: [new TextRun({ text: fixText(b.text) + '.', font: FONT, size: S.body, italics: true })] })]); i++; continue; }
  if (b.type === 'p') { push(body, 'flow', [para(b.runs)]); i++; continue; }
  if (b.type === 'tcap') {
    const n = capNum(b.text), tb = B[i + 1];
    const ncol = tb.rows[0].length, chars = Array.from({ length: ncol }, (_, j) => Math.max(...tb.rows.map((r) => String(r[j]).length))).reduce((a, c) => a + c, 0);
    const wide = TWO ? (MAIN_TABS.has(n) ? (ncol >= 5 || chars > 60) : false) : true;
    const items = [...tabCaption(n, stripCaption(b.text)), tableBlock(tb.rows, wide || !TWO ? (TWO ? TEXT_W : TEXT_W) : COL_W), new Paragraph({ spacing: { after: 100 }, children: [] })];
    push(MAIN_TABS.has(n) ? body : appendix, wide ? 'wide' : 'flow', items);
    i += 2; continue;
  }
  if (b.type === 'img') {
    const cp = B[i + 1], n = capNum(cp.text), file = b.file;
    const wide = WIDE_FIG.has(file);
    let w = TWO ? (wide ? TEXT_W : COL_W) : (wide ? TEXT_W : Math.floor(TEXT_W * 0.7));
    if (!TWO && (file === 'architecture.png' || file === 'architecture_nomod.png')) w = TEXT_W;
    if (TWO && (file === 'architecture.png' || file === 'architecture_nomod.png')) w = TEXT_W;
    const items = [imageBlock(file, w), figCaption(n, stripCaption(cp.text))];
    push(MAIN_FIGS.has(n) ? body : appendix, wide && TWO ? 'wide' : 'flow', items);
    i += 2; continue;
  }
  if (b.type === 'algo') { algoNo += 1; push(body, 'flow', [...algoParas(algoNo, fixText(b.title), b.lines, S.algo), new Paragraph({ spacing: { after: 100 }, children: [] })]); i++; continue; }
  if (b.type === 'algopair') {
    const n1 = algoNo + 1, n2 = algoNo + 2; algoNo += 2;
    push(body, 'flow', [...algoParas(n1, fixText(b.a.title), b.a.lines, S.algo), new Paragraph({ spacing: { after: 100 }, children: [] }), ...algoParas(n2, fixText(b.b.title), b.b.lines, S.algo), new Paragraph({ spacing: { after: 100 }, children: [] })]);
    i++; continue;
  }
  i++;
}

// ---- title block, abstract, references ----
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 }, children: [new TextRun({ text: 'LexFront: A Compiler-Style Lexer Used as a Pre-Tokenizer for Noisy English Text Classification', font: FONT, size: S.title })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: 'Veer Jahaan Arora (24BCB0080), Yash Aggarwal (24BKT0013), and V C Pavitran (24BCE2019)', font: FONT, size: S.auth })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: 'School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore, India', font: FONT, size: S.auth - 2, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: 'Subject: Compiler Design Lab (BCSE307P)   Faculty: Suganthini C', font: FONT, size: S.auth - 2, italics: true })] }),
];
const ABSTRACT = 'Text from social media is full of hashtags, mentions, links, emoticons and stretched-out words, and most tokenizers were not designed for it. We ask whether a lexical analyzer built the way a compiler builds one can do better. We built LexFront, a lexer with eleven token classes, longest-match scanning, error recovery, a symbol table and six switchable normalization rules, together with our own engine that turns regular expressions into an NFA, a DFA and a minimized DFA, checked against the PLY-based lexer that the experiments use. LexFront was compared with whitespace, NLTK and TweetTokenizer, and with BPE, WordPiece and Unigram subword tokenizers, with the classifier held fixed, on three core datasets, a collection of eight cleaned datasets, and four large datasets (Sentiment140, Civil Comments, Amazon polarity and Yelp). The result is consistent: LexFront is competitive but not better. With the tokenizer as the only change it stays within 0.00 to +0.46 accuracy points of NLTK on the large datasets, and it is clearly ahead only of whitespace splitting. It gives the lowest out-of-vocabulary rate and the smallest vocabulary among the word-level tokenizers. Most of the measured accuracy comes from the classifier and its tuning: moving from Naive Bayes to Logistic Regression is worth up to 17 points, and wider validated settings add 2 to 3 points. Subword tokenizers and a character n-gram baseline did not help. We report confidence intervals and the hypotheses that did not hold.';
const absParas = [
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 }, children: [
    new TextRun({ text: 'Abstract—', font: FONT, size: S.abs, bold: true, italics: true }), new TextRun({ text: fixText(ABSTRACT), font: FONT, size: S.abs, bold: true, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [
    new TextRun({ text: 'Index Terms—', font: FONT, size: S.abs, bold: true, italics: true }), new TextRun({ text: keywords, font: FONT, size: S.abs, italics: true })] }),
];
const refParas = [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: 'REFERENCES', font: FONT, size: S.h })] }),
  ...D.refs.map((r, k) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 30, line: 228 }, indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: `[${k + 1}]`, font: FONT, size: S.ref }), new TextRun({ children: [new Tab()] }),
      new TextRun({ text: r.replace(/(pp\. \d+)-(\d+)/, '$1–$2').replace(/"([^"]+)"/g, '“$1”'), font: FONT, size: S.ref })],
    tabStops: [{ type: 'left', position: 360 }],
  }))];

// appendix heading + items
const appHead = [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 0 }, children: [new TextRun({ text: 'APPENDIX A', font: FONT, size: S.h })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 80 }, children: [new TextRun({ text: 'ADDITIONAL RESULTS', font: FONT, size: S.h - 1 })] })];

// ---- assemble sections ----
const pageProps = { page: { size: { width: PAGE_W, height: PAGE_H }, margin: MARG } };
const sections = [];
const oneCol = (children, type) => ({ properties: { ...pageProps, type: type || SectionType.CONTINUOUS }, children });
const twoCol = (children) => ({ properties: { ...pageProps, type: SectionType.CONTINUOUS, column: { count: 2, space: GAP, equalWidth: true } }, children });

sections.push({ properties: { ...pageProps }, children: titleBlock });
if (TWO) {
  let flow = [...absParas];
  const flush = () => { if (flow.length) { sections.push(twoCol(flow)); flow = []; } };
  const all = [...body, { kind: 'flow', items: appHead }, ...appendix];
  for (const blk of all) {
    if (blk.kind === 'wide') { flush(); sections.push(oneCol(blk.items)); }
    else flow.push(...blk.items);
  }
  flow.push(...refParas);
  flush();
} else {
  const kids = [...absParas.map((p) => p)];
  for (const blk of [...body, { kind: 'flow', items: appHead }, ...appendix]) kids.push(...blk.items);
  kids.push(...refParas);
  sections.push(oneCol(kids));
}

const doc = new Document({
  creator: 'LexFront team', title: 'LexFront manuscript',
  styles: { default: { document: { run: { font: FONT, size: S.body } } } },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT, 'sections', sections.length); });
