'use strict';
const $ = (id) => document.getElementById(id);
const tags = {bho:'bho_Deva', mai:'mai_Deva', mag:'mag_Deva', hin:'hin_Deva', eng:'eng_Latn'};
let commands = '', generatedConfig, configName;
function updateExperiment() {
  const focus = $('language').value, bridge = $('bridge').value;
  const [src, tgt] = $('direction').value === 'to' ? [focus, bridge] : [bridge, focus];
  const mode = $('mode').value, seed = Number($('seed').value), pair = `${src}-${tgt}`;
  const run = `runs/${pair}-${mode}-s${seed}`;
  configName = `nllb-${pair}-${mode}-s${seed}.json`;
  generatedConfig = {model:'facebook/nllb-200-distilled-600M', revision:null, src_lang:tags[src], tgt_lang:tags[tgt],
    train_file:`data/local/${pair}/train.jsonl`, dev_file:`data/local/${pair}/dev.jsonl`, output_dir:run,
    mode, seed, epochs:3, learning_rate:mode === 'lora' ? 0.0002 : 0.00002, batch_size:1,
    gradient_accumulation_steps:16, max_length:128, precision:'fp16', gradient_checkpointing:true, lora_rank:8};
  const audit = `bhaashik audit \\\n  --train data/local/${pair}/train.jsonl \\\n  --dev data/local/${pair}/dev.jsonl \\\n  --test data/local/${pair}/test.jsonl`;
  const train = `# Save the downloaded config in configs/\nbhaashik train --config configs/${configName}`;
  const evaluate = `bhaashik translate --model ${run} \\\n  --src ${tags[src]} --tgt ${tags[tgt]} \\\n  --input data/local/${pair}/test.jsonl \\\n  --output ${run}/predictions.jsonl\nbhaashik evaluate \\\n  --input ${run}/predictions.jsonl \\\n  --output ${run}/evaluation.json\nbhaashik errors \\\n  --input ${run}/predictions.jsonl \\\n  --output ${run}/annotations.csv`;
  $('audit-command').textContent = audit;
  $('train-command').textContent = train;
  $('eval-command').textContent = evaluate;
  commands = `${audit}\n\n${train}\n\n${evaluate}\n`;
  $('budget-note').textContent = mode === 'lora'
    ? 'Start with batch 1, length 128 and gradient checkpointing. GPU memory depends on the runtime.'
    : 'Full finetuning updates all weights and uses more memory. A free notebook GPU may not fit this budget.';
}
$('experiment-form').addEventListener('change', updateExperiment);
$('experiment-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const url = URL.createObjectURL(new Blob([JSON.stringify(generatedConfig, null, 2)+'\n'], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = configName; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('status').textContent = 'Configuration downloaded. Save it in the repository configs directory.';
});
$('copy-commands').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(commands); $('copy-commands').textContent = 'Copied ✓'; }
  catch { $('copy-commands').textContent = 'Select commands to copy'; }
  setTimeout(() => { $('copy-commands').textContent = 'Copy commands'; }, 2500);
});
function validScore(v) { return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 100.000001; }
function validateReport(r) {
  if (!r || r.schema_version !== 1 || !Number.isInteger(r.n) || r.n < 1 ||
      !validScore(r.scores?.bleu) || !validScore(r.scores?.chrfpp) ||
      typeof r.scores?.signatures?.bleu !== 'string' || typeof r.scores?.signatures?.chrfpp !== 'string') {
    throw new Error('Expected a schema v1 report from bhaashik evaluate with valid scores and signatures.');
  }
  if (r.paired_bootstrap) {
    for (const key of ['bleu','chrfpp']) {
      const delta = r.paired_bootstrap.delta_candidate_minus_baseline?.[key];
      if (!delta || !Number.isFinite(delta.point) || !Array.isArray(delta.ci95) || delta.ci95.length !== 2 ||
          !delta.ci95.every(Number.isFinite) || delta.ci95[0] > delta.ci95[1]) throw new Error('Invalid paired confidence interval.');
    }
  }
}
$('report-file').addEventListener('change', async (e) => {
  $('report-error').textContent = '';
  const file = e.target.files[0]; if (!file) return;
  try {
    if (file.size > 5 * 1024 * 1024) throw new Error('Please choose a JSON report smaller than 5 MB.');
    const r = JSON.parse(await file.text()); validateReport(r);
    $('bleu-score').textContent = r.scores.bleu.toFixed(2);
    $('chrf-score').textContent = r.scores.chrfpp.toFixed(2);
    $('sample-count').textContent = r.n.toLocaleString();
    $('report-state').textContent = file.name;
    if (r.paired_bootstrap) {
      const delta = r.paired_bootstrap.delta_candidate_minus_baseline.chrfpp;
      $('comparison').textContent = `chrF++ Δ vs baseline: ${delta.point.toFixed(2)} · 95% CI [${delta.ci95.map(v => v.toFixed(2)).join(', ')}]. Resampling unit: ${r.paired_bootstrap.unit}.`;
    } else { $('comparison').textContent = 'Report loaded. No paired baseline comparison was included.'; }
    $('report-details').hidden = false;
    $('report-metadata').textContent = JSON.stringify({signatures:r.scores.signatures, domains:r.domains, input_sha256:r.input_sha256}, null, 2);
    $('status').textContent = `Loaded evaluation for ${r.n} test sentences.`;
  } catch (err) {
    $('report-error').textContent = err.message;
    $('report-state').textContent = 'Invalid report';
    for (const id of ['bleu-score','chrf-score','sample-count']) $(id).textContent = '—';
    $('comparison').textContent = 'Choose a valid report to display results.';
    $('report-details').hidden = true;
  }
});
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) if (entry.isIntersecting) {
    document.querySelectorAll('nav a').forEach(a => {
      const active = a.hash === '#'+entry.target.id;
      a.classList.toggle('active', active);
      if (active) a.setAttribute('aria-current','location'); else a.removeAttribute('aria-current');
    });
  }
}, {rootMargin:'-5% 0px -60% 0px'});
document.querySelectorAll('main section[id]').forEach(s => observer.observe(s));
updateExperiment();
