// Runs the engine block of the shipped page against Python reference cases.
// Usage: node tests/test_js_parity.js [docs/index.html]
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(process.argv[2] || path.join(root, 'web', 'site_template.html'), 'utf8');
const B = JSON.parse(fs.readFileSync(path.join(root, 'web', 'bundle.json'), 'utf8'));
const start = html.indexOf('/* ---------- engine');
const end = html.indexOf('/* ---------- static numbers');
if (start < 0 || end < 0) throw new Error('engine block not found');
const src = 'const M=B.model, D=M.design; const sig=z=>1/(1+Math.exp(-z));' + html.slice(start, end) + 'return {eta, pInt};';
const {eta, pInt} = new Function('B', src)(B);
let worst = 0;
for (const c of B.parity) {
  const e = eta(c.home), p = pInt(e);
  worst = Math.max(worst, Math.abs(e - c.eta), Math.abs(p - c.p_prior));
}
console.log(`parity cases ${B.parity.length}, max abs difference ${worst.toExponential(2)}`);
if (worst > 1e-6) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
