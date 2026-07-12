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
 
  submitBtn.disabled = true;
  submitBtn.textContent = 'Thinking...';
 
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    answerText.textContent = 'Error: ' + err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Ask';
  }
});
 
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
 
  answerText.textContent = data.answer_text;
 
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
