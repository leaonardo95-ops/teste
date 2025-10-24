let model;

// 🧠 Banco de perguntas e respostas
const perguntas = {
  1: "Quais faixas de renda são isentas?",
  2: "A isenção vale para todos os tipos de renda?",
  3: "Quem é MEI ou autônomo também tem direito à isenção?"
};

const respostas = {
  1: "Rendas de até R$5.000,00 serão isentas.",
  2: "A PL trata 'rendimentos' em termos gerais.",
  3: "Sim, a isenção vale para todos os tipos de rendas tributáveis."
};

// 🚀 Inicialização
async function init() {
  try {
    console.log("Carregando modelo...");
    model = await use.load();
    console.log("Modelo carregado com sucesso!");

    // Agora que o modelo carregou, habilita o botão
    const btn = document.getElementById("enviar");
    btn.addEventListener("click", responderPergunta);
  } catch (err) {
    console.error("Erro ao carregar o modelo:", err);
  }
}

// 🔍 Função para comparar similaridade semântica
async function calcularSimilaridade(texto1, texto2) {
  const embeddings = await model.embed([texto1, texto2]);
  const vecs = await embeddings.array();
  const [a, b] = vecs;

  // produto escalar
  const dot = a.map((v, i) => v * b[i]).reduce((acc, val) => acc + val, 0);
  // norma vetorial
  const normA = Math.sqrt(a.reduce((acc, val) => acc + val ** 2, 0));
  const normB = Math.sqrt(b.reduce((acc, val) => acc + val ** 2, 0));

  return dot / (normA * normB);
}

// 💬 Função principal de resposta
async function responderPergunta() {
  const texto = document.getElementById("pergunta").value.trim();
  const respDiv = document.getElementById("resposta");

  if (!texto) {
    respDiv.textContent = "Por favor, digite uma pergunta.";
    return;
  }

  let melhorIndice = null;
  let maiorSimilaridade = -1;

  for (const [i, pergunta] of Object.entries(perguntas)) {
    const similaridade = await calcularSimilaridade(texto, pergunta);
    console.log(`Similaridade [${i}]:`, similaridade);

    if (similaridade > maiorSimilaridade) {
      maiorSimilaridade = similaridade;
      melhorIndice = i;
    }
  }

  // Classificação simples
  if (maiorSimilaridade > 0.75) {
    respDiv.textContent = respostas[melhorIndice];
  } else if (maiorSimilaridade < 0.4) {
    respDiv.textContent = "Desculpe, não tenho certeza sobre isso. Pode reformular sua pergunta?";
  } else {
    respDiv.textContent = "Poderia especificar melhor sua pergunta?";
  }
}

// 🔁 Quando a página carregar, inicia o modelo
window.onload = init;
