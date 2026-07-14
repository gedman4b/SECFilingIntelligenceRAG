const questionInput = document.getElementById('question');
const submitBtn = document.getElementById('submit');
const responseSection = document.getElementById('response');
const confidenceBadge = document.getElementById('confidence');
const answerText = document.getElementById('answer-text');
const calculationBlock = document.getElementById('calculation-block');
const warningsBlock = document.getElementById('warnings-block');
const citationsBlock = document.getElementById('citations-block');
 
submitBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  clearResponse();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Thinking...';

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      throw new Error(`Backend returned HTTP ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    responseSection.classList.remove('hidden');
    confidenceBadge.className = 'confidence-badge conf-insufficient';
    confidenceBadge.textContent = 'ERROR';
    answerText.textContent = 'Error: ' + err.message;
    calculationBlock.innerHTML = '';
    warningsBlock.innerHTML = '';
    citationsBlock.innerHTML = '';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Ask';
  }
});
 
function clearResponse() {
  responseSection.classList.add('hidden');
  confidenceBadge.className = 'confidence-badge';
  confidenceBadge.textContent = '';
  answerText.textContent = '';
  calculationBlock.innerHTML = '';
  warningsBlock.innerHTML = '';
  citationsBlock.innerHTML = '';
}

function renderResponse(data) {
  responseSection.classList.remove('hidden');
  const confMap = {
    high: { text: 'HIGH CONFIDENCE', cls: 'conf-high' },
    medium: { text: 'MEDIUM CONFIDENCE', cls: 'conf-medium' },
    low: { text: 'LOW CONFIDENCE', cls: 'conf-low' },
    insufficient_data: { text: 'INSUFFICIENT DATA', cls: 'conf-insufficient' },
  };
  const c = confMap[data.confidence] || confMap.low;
  confidenceBadge.className = 'confidence-badge ' + c.cls;
  confidenceBadge.textContent = c.text;
 
  answerText.innerHTML = renderMarkdownLite(data.answer_text);
 
  // Calculation block
  if (data.computation_expression) {
    calculationBlock.innerHTML =
      '<h3>Calculation</h3>' +
      '<pre class="calc-expr">' + escapeHtml(data.computation_expression) + '</pre>';
  } else {
    calculationBlock.innerHTML = '';
  }
 
  // Warnings block: severity-styled
  if (data.warnings && data.warnings.length) {
    warningsBlock.innerHTML =
      '<h3>Warnings and notes</h3>' +
      data.warnings.map(w =>
        `<div class="warning warning-${w.severity}">` +
        `<span class="warning-severity">${w.severity.toUpperCase()}</span>` +
        `<span class="warning-message">${escapeHtml(w.message)}</span>` +
        '</div>'
      ).join('');
  } else {
    warningsBlock.innerHTML = '';
  }
 
  // Citations block: table of raw values with clickable filing links
  if (data.citations && data.citations.length) {
    citationsBlock.innerHTML =
      '<table class="citations-table">' +
      '<thead><tr><th>Label</th><th>Value</th><th>Period</th><th>Source</th></tr></thead>' +
      '<tbody>' +
      data.citations.map(cite =>
        '<tr>' +
        `<td>${escapeHtml(cite.label)}</td>` +
        `<td class="num-cell">${cite.value.toLocaleString()} ${escapeHtml(cite.units)}</td>` +
        `<td>${escapeHtml(cite.period)}</td>` +
        `<td><a href="${escapeHtml(cite.filing_url)}#page=${cite.page}" target="_blank">` +
        `Filing page ${cite.page}</a></td>` +
        '</tr>'
      ).join('') +
      '</tbody></table>';
  } else {
    citationsBlock.innerHTML = '<p>No numeric facts retrieved for this answer.</p>';
  }
}
 
function escapeHtml(s) {
  if (typeof s !== 'string') s = String(s);
  return s.replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

// The Answer Composer's narrative is Markdown-ish prose (headers, bold,
// bullet lists). Escapes first, then recognizes only that small subset
// against the already-escaped text, so no LLM output is ever interpreted
// as HTML: a stray "<script>" in an answer renders as the literal text
// "<script>", never as a tag.
function renderMarkdownLite(raw) {
  const lines = escapeHtml(raw).split('\n');
  const htmlParts = [];
  let paragraphBuffer = [];
  let listBuffer = [];

  function flushParagraph() {
    if (paragraphBuffer.length) {
      htmlParts.push('<p>' + paragraphBuffer.join(' ') + '</p>');
      paragraphBuffer = [];
    }
  }
  function flushList() {
    if (listBuffer.length) {
      htmlParts.push('<ul>' + listBuffer.map(item => `<li>${item}</li>`).join('') + '</ul>');
      listBuffer = [];
    }
  }
  function inlineFormat(s) {
    return s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const headerMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);

    if (headerMatch) {
      flushParagraph();
      flushList();
      const level = headerMatch[1].length + 2; // # -> h3, ## -> h4, ### -> h5
      htmlParts.push(`<h${level}>${inlineFormat(headerMatch[2])}</h${level}>`);
    } else if (bulletMatch) {
      flushParagraph();
      listBuffer.push(inlineFormat(bulletMatch[1]));
    } else if (trimmed === '') {
      flushParagraph();
      flushList();
    } else {
      flushList();
      paragraphBuffer.push(inlineFormat(trimmed));
    }
  }
  flushParagraph();
  flushList();

  return htmlParts.join('');
}
