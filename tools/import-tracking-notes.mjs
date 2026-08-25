import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const notesRoot = process.argv[2];
const requestedSlugs = new Set(process.argv.slice(3));
if (!notesRoot) {
  console.error('Usage: node tools/import-tracking-notes.mjs <tracking-notes-directory>');
  process.exit(1);
}

const repoRoot = path.resolve(import.meta.dirname, '..');
const postsRoot = path.join(repoRoot, 'source', '_posts');
const noteCollection = path.basename(path.resolve(notesRoot));
const collectionConfig = {
  Tracking: { category: 'Tracking', imageDirectory: 'tracking' },
  IR_VIS_Reg: { category: '红外-可见光配准', imageDirectory: 'ir-vis-reg' },
}[noteCollection] ?? {
  category: noteCollection.replaceAll('_', ' '),
  imageDirectory: noteCollection.toLowerCase().replaceAll('_', '-'),
};
const imageRoot = path.join(repoRoot, 'source', 'images', collectionConfig.imageDirectory);
const publicImageBase = `/images/${collectionConfig.imageDirectory}`;

const yamlQuote = (value) => JSON.stringify(String(value));

function parseSourceFrontMatter(markdown) {
  const match = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!match) return { body: markdown, raw: '' };
  return { body: markdown.slice(match[0].length), raw: match[1] };
}

function scalar(frontMatter, key) {
  return frontMatter.match(new RegExp(`^${key}:\\s*(.+?)\\s*$`, 'm'))?.[1]
    ?.replace(/\s+#.*$/, '')
    .trim();
}

function list(frontMatter, key) {
  const match = frontMatter.match(new RegExp(`^${key}:\\s*\\r?\\n((?:\\s+-\\s+.*\\r?\\n?)+)`, 'm'));
  if (!match) return [];
  return [...match[1].matchAll(/^\s+-\s+(.+?)\s*$/gm)].map((item) => item[1]);
}

function plainText(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`>#|]/g, ' ')
    .replace(/\$+[^$]*\$+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function cleanInlineTitle(value) {
  return plainText(value.replace(/<br\s*\/?\s*>/gi, ' ').replace(/<[^>]+>/g, ' '));
}

function extractAbstract(body) {
  const match = body.match(/^## Abstract(?:（[^\r\n]*）)?\s*\r?\n([\s\S]*?)(?=\r?\n---\s*$)/m);
  if (!match) return '';
  const text = plainText(match[1]);
  if (text.length <= 220) return text;
  const candidate = text.slice(0, 220);
  const sentenceEnd = Math.max(
    candidate.lastIndexOf('。'),
    candidate.lastIndexOf('；'),
    candidate.lastIndexOf('！'),
    candidate.lastIndexOf('？'),
  );
  return `${candidate.slice(0, sentenceEnd >= 120 ? sentenceEnd + 1 : 217).trimEnd()}…`;
}

function normalizeHeadings(body) {
  const mainHeadings = new Map([
    ['📖 论文信息', '论文信息'],
    ['🚀 快速导航', '论文资源'],
    ['🧠 1. Motivation', '1. 研究动机'],
    ['💡 2. Contributions', '2. 主要贡献'],
    ['🏗️ 3. Method', '3. 方法'],
    ['🧪 4. Experiments', '4. 实验'],
    ['🔬 5. Reproduction', '5. 复现指南'],
    ['🤔 6. Critical Thinking', '6. 批判性思考'],
    ['📝 7. 深度阅读标注', '7. 深度阅读标注'],
    ['🎯 8. Final Takeaway', '8. 总结'],
  ]);
  const translated = new Map([
    ['Abstract', '摘要'],
    ['Overall Pipeline', '整体框架'],
    ['Paper ↔ Code', '论文与代码对照'],
    ['Training / Inference', '训练与推理'],
    ['Datasets & Metrics', '数据集与指标'],
    ['Main Results', '主要结果'],
    ['Ablation', '消融实验'],
    ['Failure Cases', '失败案例'],
    ['Code', '代码对应'],
    ['Sources', '参考资料'],
    ['Final Takeaway', '总结'],
  ]);

  let inFence = false;
  return body
    .split(/\r?\n/)
    .map((line) => {
      if (/^\s*```/.test(line)) {
        inFence = !inFence;
        return line;
      }
      if (inFence) return line;
      const heading = line.match(/^(#{1,5})\s+(.+?)\s*$/);
      if (!heading) return line;
      const [, marks, rawTitle] = heading;
      if (marks === '#' && mainHeadings.has(rawTitle)) {
        return `## ${mainHeadings.get(rawTitle)}`;
      }
      const level = marks === '#' ? 3 : Math.min(marks.length + 1, 6);
      const normalizedTitle = rawTitle
        .replace(/^(\d+(?:\.\d+)?) Overall Pipeline$/, '$1 整体框架')
        .replace(/^(\d+(?:\.\d+)?) Paper ↔ Code$/, '$1 论文与代码对照')
        .replace(/^(\d+(?:\.\d+)?) Training \/ Inference$/, '$1 训练与推理');
      return `${'#'.repeat(level)} ${translated.get(normalizedTitle) ?? normalizedTitle}`;
    })
    .join('\n');
}

const inlineMathReplacements = new Map([
  ['[Ex; Esz; Edz]', '[E_x; E_{sz}; E_{dz}]'],
  ['E ∈ R^{N×2D}', 'E \\in \\mathbb{R}^{N \\times 2D}'],
  ['Et', 'E_t'],
  ['Etext', 'E_{\\text{text}}'],
  ['M(i,j)=1 if (i,j) inside B', 'M(i,j)=1 \\text{ if } (i,j) \\in B'],
  ['QdzKx^T', 'Q_{dz}K_x^T'],
  ["Qsz'", "Q_{sz}'"],
  ['QszKdz^T', 'Q_{sz}K_{dz}^T'],
  ['QszKx^T', 'Q_{sz}K_x^T'],
  ['QtKx^T', 'Q_tK_x^T'],
  ['QxKt^T', 'Q_xK_t^T'],
  ['QxKx^T', 'Q_xK_x^T'],
  ['ωdz', '\\omega_{dz}'],
  ["ωdz = softmax(Qsz'Kdz^T/√dk)", "\\omega_{dz} = \\operatorname{softmax}(Q_{sz}'K_{dz}^T/\\sqrt{d_k})"],
  ['ωsz', '\\omega_{sz}'],
  ["ωsz = softmax(Qsz'Ksz^T/√dk)", "\\omega_{sz} = \\operatorname{softmax}(Q_{sz}'K_{sz}^T/\\sqrt{d_k})"],
  ['ωx', '\\omega_x'],
  ["ωx = softmax(Qsz'Kx^T/√dk)", "\\omega_x = \\operatorname{softmax}(Q_{sz}'K_x^T/\\sqrt{d_k})"],
  [
    "ω = softmax(Qsz'K^T) + softmax(Qdt-mean K^T)",
    "\\omega = \\operatorname{softmax}(Q_{sz}'K^T) + \\operatorname{softmax}(Q_{\\mathrm{dt\\text{-}mean}}K^T)",
  ],
]);

function normalizeInlineMath(body) {
  return body.replace(/`([^`\r\n]+)`/g, (match, content) => {
    const formula = inlineMathReplacements.get(content);
    if (formula) return `$${formula}$`;

    const value = content.trim();
    const isSingleSymbol = /^[A-Za-zΑ-ω]$/u.test(value);
    const isIndexedSymbol = /^[A-Za-zΑ-ω]\d+$/u.test(value);
    const isSymbolList = /^(?:[A-Za-zΑ-ω](?:_[A-Za-z0-9]+|\d+)?)(?:\s*,\s*[A-Za-zΑ-ω](?:_[A-Za-z0-9]+|\d+)?)+$/u.test(value);
    const hasMathNotation = /[_^=<>∈≤≥≈≪∞∼λμωστθδΔΓΣ⊙⊗⊕×+]/u.test(value);
    const startsLikeVariable = /^[A-Za-zΑ-ω](?:['’])?(?:[_^0-9{(]|\s*[=<>∈≤≥≈≪])/u.test(value);
    const isLatexFragment = /^\\(?:bar|hat|tilde|Sigma|Delta|Gamma|lambda|mu|omega|sigma|tau|theta|delta|odot|otimes|oplus|in)\b/.test(value);
    const looksLikePathOrCode = /(?:https?:\/\/|[\\/].+\.(?:py|json|txt|md|pdf)|\.(?:py|json|txt|md|pdf)$|->$|\b(?:Conv|ReLU|Linear|Pool|Softplus|TopK)\b)/i.test(value);

    if (looksLikePathOrCode || !(isSingleSymbol || isIndexedSymbol || isSymbolList || isLatexFragment || (hasMathNotation && startsLikeVariable))) {
      return match;
    }

    const normalized = value.replace(/_([A-Za-z][A-Za-z0-9]{1,})(?=$|[\s,()/\[\].+*=<>^])/g, '_{\\mathrm{$1}}');
    return `$${normalized}$`;
  });
}

function polishBody(body, slug) {
  body = body.replace(/\r\n/g, '\n');
  body = body.replace(/^# 📥 增量导入记录[\s\S]*$/m, '');
  body = body
    .split('\n')
    .filter((line) => !/zotero:\/\//i.test(line))
    .filter((line) => !/^\*\*Zotero:\*\*/i.test(line))
    .filter((line) => !/^[-*]\s+\*\*本地素材[：:]\*\*/.test(line))
    .filter((line) => !/^\*\*[^:]+:\*\*\s*—(?:\s*\|\s*—)?\s*$/.test(line))
    .join('\n');
  body = body.replace(/^> \[!important\](?:\s+(.+))?\s*$/gm, (_match, title) =>
    title ? `> **阅读说明｜${title.trim()}**` : '> **阅读说明**',
  );
  body = body.replace(/^> \[!warning\](?:\s+(.+))?\s*$/gm, (_match, title) =>
    title ? `> **注意｜${title.trim()}**` : '> **注意**',
  );
  body = body.replace(/^> \[!note\](?:\s+(.+))?\s*$/gm, (_match, title) =>
    title ? `> **补充说明｜${title.trim()}**` : '> **补充说明**',
  );
  body = body.replace(
    /有官方代码时，Method 必须结合源码理解；没有代码时按照论文 Method 和 Supplementary 整理。/g,
    '方法部分优先结合公开源码理解；未提供代码时，则依据论文与补充材料整理。',
  );
  body = body.replace(/^暂无 Zotero 标注.*$/gm, '本节暂无额外阅读标注。');
  body = body.replace(/^本地素材:\s*Zotero 条目（全文已提取为 .*?）\s*$/gm, '材料说明：本文依据论文全文整理。');
  body = body.replace(/^- \*\*Local Code:\*\*.*\r?\n?/gm, '');
  body = body.replace(/^Local:\s*[A-Za-z]:[\\/].*\r?\n?/gm, '');
  body = body.replace(/基于论文全文（`?\.papers_fulltext\/[^)`]+`?）/g, '基于论文全文');
  body = body.replace(/本地没有 Zotero 高亮；基于 full text 的深读标注/g, '本文未包含额外高亮；以下为基于论文全文的深读标注');
  body = body.replace(/没有本地代码，不能声称复现论文数值/g, '论文未提供公开代码，不能声称复现论文数值');
  body = normalizeInlineMath(body);
  body = body.replace(/\n## 论文图示（截图）\s*\n\s*## 论文图示（截图）/g, '\n## 论文图示（截图）');
  body = body.replace(
    /!\[([^\n]*?)\]\(assets\/([^/]+)\/([^)]+?)\.(?:png|jpg|jpeg)\)/gi,
    (_match, alt, assetSlug, filename) =>
      `![${alt}](${publicImageBase}/${assetSlug}/${filename}.webp)`,
  );
  // Keep display math in a single Markdown text node. Otherwise Markdown turns
  // physical newlines into <br> elements before the math renderer can see it.
  body = body.replace(/\$\$([\s\S]*?)\$\$/g, (_match, formula) => {
    const normalizedFormula = formula.replace(/\s*\n\s*/g, ' ').trim();
    return `$$${normalizedFormula}$$`;
  });
  body = body.replace(
    /(## Abstract(?:（[^\n]*）)?\s*\n[\s\S]*?\n)(---\s*\n)/,
    '$1<!-- more -->\n\n$2',
  );
  body = normalizeHeadings(body);
  body = body.replace(/\n{4,}/g, '\n\n\n').trim();

  const editorialNote = [
    '> 本文基于论文、补充材料与公开代码整理。文中的“我的理解”和“批判性思考”属于个人分析；',
    '> 论文插图均来自原论文或补充材料，仅用于学习与讨论。',
  ].join('\n');

  return `${editorialNote}\n\n${body}\n`;
}

function referencedImages(body) {
  return [...body.matchAll(/!\[[^\n]*?\]\((assets\/[^)]+?\.(?:png|jpg|jpeg))\)/gi)].map(
    (match) => match[1],
  );
}

function convertImage(source, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (
    fs.existsSync(destination) &&
    fs.statSync(destination).mtimeMs >= fs.statSync(source).mtimeMs
  ) {
    return;
  }
  const result = spawnSync(
    'ffmpeg',
    [
      '-y',
      '-loglevel',
      'error',
      '-i',
      source,
      '-c:v',
      'libwebp',
      '-quality',
      '86',
      '-compression_level',
      '6',
      destination,
    ],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    throw new Error(`Failed to convert ${source}: ${result.stderr || result.stdout}`);
  }
}

fs.mkdirSync(postsRoot, { recursive: true });
fs.mkdirSync(imageRoot, { recursive: true });

const noteFiles = fs
  .readdirSync(notesRoot)
  .filter((name) => name.endsWith('_笔记.md'))
  .filter((name) => requestedSlugs.size === 0 || requestedSlugs.has(name.replace(/_笔记\.md$/, '')))
  .sort((a, b) => a.localeCompare(b, 'en'));

const dayCounters = new Map();
const report = [];
const expectedPosts = new Set();
let originalImageBytes = 0;
let outputImageBytes = 0;

for (const filename of noteFiles) {
  const slug = filename.replace(/_笔记\.md$/, '');
  const inputPath = path.join(notesRoot, filename);
  const markdown = fs.readFileSync(inputPath, 'utf8');
  const { body, raw } = parseSourceFrontMatter(markdown);
  const rawPaperTitle = body.match(/^\*\*Title:\*\*\s*(.+?)\s*$/m)?.[1]?.trim();
  const paperTitle = rawPaperTitle ? cleanInlineTitle(rawPaperTitle) : slug;
  const updated = scalar(raw, 'updated') ?? scalar(raw, 'date') ?? '2026-08-18';
  const countForDay = dayCounters.get(updated) ?? 0;
  dayCounters.set(updated, countForDay + 1);
  const hour = 20 + Math.floor(countForDay / 12);
  const minute = (countForDay % 12) * 5;
  const publicationTime = `${updated} ${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00`;
  const originalTags = list(raw, 'tags');
  const task = scalar(raw, 'task');
  const tags = [...new Set([...originalTags, task, collectionConfig.category].filter(Boolean))];
  const description = extractAbstract(body);

  const frontMatterLines = [
    '---',
    `title: ${yamlQuote(`论文阅读｜${paperTitle}`)}`,
    'categories:',
    '  - 文献阅读',
    `  - ${yamlQuote(collectionConfig.category)}`,
    'tags:',
    ...tags.map((tag) => `  - ${yamlQuote(tag)}`),
    `description: ${yamlQuote(description)}`,
    'readmore: true',
    'mathjax: true',
    `date: ${publicationTime}`,
    `updated: ${updated} 23:00:00`,
    '---',
    '',
  ];

  const outputName = `论文阅读-${slug}.md`;
  const outputPath = path.join(postsRoot, outputName);
  const existingMarkdown = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : '';
  const existingAbbrlink = scalar(parseSourceFrontMatter(existingMarkdown).raw, 'abbrlink');
  if (existingAbbrlink) frontMatterLines.splice(-2, 0, `abbrlink: ${yamlQuote(existingAbbrlink.replace(/^['"]|['"]$/g, ''))}`);
  const frontMatter = frontMatterLines.join('\n');
  expectedPosts.add(outputPath);
  fs.writeFileSync(outputPath, `${frontMatter}${polishBody(body, slug)}`, 'utf8');

  const imageRefs = [...new Set(referencedImages(body))];
  for (const ref of imageRefs) {
    const source = path.resolve(notesRoot, ref);
    if (!source.startsWith(path.resolve(notesRoot) + path.sep) || !fs.existsSync(source)) {
      throw new Error(`Invalid or missing image reference in ${filename}: ${ref}`);
    }
    const relative = path.relative(path.join(notesRoot, 'assets'), source);
    const destination = path.join(imageRoot, relative).replace(/\.(?:png|jpg|jpeg)$/i, '.webp');
    convertImage(source, destination);
    originalImageBytes += fs.statSync(source).size;
    outputImageBytes += fs.statSync(destination).size;
  }

  report.push({ slug, outputName, title: paperTitle, images: imageRefs.length });
}

if (requestedSlugs.size === 0) {
  for (const entry of fs.readdirSync(postsRoot)) {
    if (!/^论文阅读-.+\.md$/.test(entry)) continue;
    const fullPath = path.join(postsRoot, entry);
    if (!expectedPosts.has(fullPath)) fs.rmSync(fullPath);
  }
}

console.log(
  JSON.stringify(
    {
      posts: report.length,
      images: report.reduce((sum, item) => sum + item.images, 0),
      originalImageMB: +(originalImageBytes / 1024 / 1024).toFixed(2),
      outputImageMB: +(outputImageBytes / 1024 / 1024).toFixed(2),
      report,
    },
    null,
    2,
  ),
);
