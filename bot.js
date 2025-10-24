// bot.js
console.log("TensorFlow:", typeof tf);
console.log("USE:", typeof use);

let model = null;
let faq = [];
let faqEmbeddings = null;

const statusEl = document.getElementById('status');
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

function addMessage(text, from='bot') {
  const d = document.createElement('div');
  d.className = 'msg ' + (from === 'user' ? 'user' : 'bot');
  d.textContent = text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function normalizeText(s) {
  return s
    .toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '') // remover acentos
    .replace(/[^\w\s]/g, ' ') // remover pontuação
    .replace(/\s+/g, ' ')
    .trim();
}

function jaccard(aTokens, bTokens) {
  const A = new Set(aTokens);
  const B = new Set(bTokens);
  const inter = new Set([...A].filter(x => B.has(x)));
  const union = new Set([...A, ...B]);
  return inter.size === 0 ? 0 : inter.size / union.size;
}

function cosine(u, v) {
  let dot = 0, nu = 0, nv = 0;
  for (let i = 0; i < u.length; i++) {
    dot += u[i] * v[i];
    nu += u[i]*u[i];
    nv += v[i]*v[i];
  }
  return dot / (Math.sqrt(nu) * Math.sqrt(nv) + 1e-8);
}

async function loadFaq() {
  const res = await fetch('faq.json');
  faq = await res.json();
}

async function embedFaqs() {
  const texts = faq.map(item => item.pergunta);
  faqEmbeddings = await model.embed(texts);
  // faqEmbeddings is a tensor; convert to array for easier use
  faqEmbeddings = await faqEmbeddings.array();
}

async function init() {
  statusEl.textContent = 'Carregando modelo... (pode demorar alguns segundos na primeira vez)';
  model = await use.load(); // universal-sentence-encoder
  statusEl.textContent = 'Carregando FAQ...';
  await loadFaq();
  statusEl.textContent = 'Criando embeddings da FAQ...';
  await embedFaqs();
  statusEl.textContent = 'Pronto — pergunte algo sobre reforma tributária!';
}

async function handleQuestion(raw) {
  const text = raw.trim();
  if (!text) return;
  addMessage(text, 'user');
  statusEl.textContent = 'Analisando...';

  const norm = normalizeText(text);
  const tokens = norm.split(' ').filter(Boolean);

  // embedding do input
  const embTensor = await model.embed([text]);
  const embArr = (await embTensor.array())[0];

  // calcular similaridades
  let bestIdx = 0;
  let bestScore = -1;
  const weights = { sem: 0.7, jac: 0.2, len: 0.1 }; // ajuste de pesos

  for (let i = 0; i < faq.length; i++) {
    const qnorm = normalizeText(faq[i].pergunta);
    const qtokens = qnorm.split(' ').filter(Boolean);

    const sem = cosine(embArr, faqEmbeddings[i]);
    const jac = jaccard(tokens, qtokens);
    // overlap coefficient: inter / min(|A|,|B|)
    const inter = tokens.filter(t => qtokens.includes(t)).length;
    const overlap = inter === 0 ? 0 : inter / Math.min(tokens.length, qtokens.length);

    const score = weights.sem*sem + weights.jac*jac + weights.len*overlap;

    if (score > bestScore) { bestScore = score; bestIdx = i; }
  }

  // thresholds — ajuste conforme necessário
  const HIGH = 0.65;
  const LOW = 0.30;

  if (bestScore >= HIGH) {
    addMessage(faq[bestIdx].resposta, 'bot');
  } else if (bestScore < LOW) {
    addMessage('Desculpe — não encontrei uma resposta segura. Pode reformular ou ser mais específico?', 'bot');
  } else {
    // resposta com incerteza: mostrar resposta candidata + pedir confirmação
    addMessage(`Talvez você queira saber: "${faq[bestIdx].pergunta}"\nResposta sugerida: ${faq[bestIdx].resposta}\nSe não, reformule sua pergunta.`, 'bot');
  }

  statusEl.textContent = `Última checagem: score=${bestScore.toFixed(2)} (modelo on-device)`;
}

sendBtn.addEventListener('click', () => {
  handleQuestion(inputEl.value);
  inputEl.value = '';
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// inicialização
let use = window.use; // universal sentence encoder é carregado globalmente
if (!use) {
  // em alguns CDN, o objeto pode vir sob diferente namespace
  // mas carregamos via <script> em index.html, então deve existir
}
init().catch(err => {
  console.error(err);
  statusEl.textContent = 'Erro ao inicializar: veja console.';
});
