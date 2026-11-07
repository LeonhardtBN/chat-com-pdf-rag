"""RAG sobre PDFs: extração por página, divisão em trechos, índice FAISS e resposta com fontes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MODELO_CHAT = os.getenv("MODELO_CHAT", "gemini-3.8-flash")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "gemini-embedding-001")
TRECHOS_POR_PERGUNTA = 4

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você responde perguntas sobre documentos em PDF, em português do Brasil.\n"
     "Use SOMENTE os trechos abaixo. Se a resposta não estiver neles, diga que não encontrou no documento.\n"
     "Ao final de cada afirmação, cite a fonte no formato [arquivo, p. N].\n\n"
     "Trechos:\n{contexto}"),
    ("placeholder", "{historico}"),
    ("human", "{pergunta}"),
])


@dataclass
class Resposta:
    texto: str
    fontes: list[Document]


def modelos_padrao():
    """Modelo de chat e de embeddings do Gemini (plano gratuito). A chave vem de GEMINI_API_KEY."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(model=MODELO_CHAT, thinking_config={"thinking_level": "low"},
                                 max_retries=2, timeout=90)
    return llm, GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)


def mensagem_de_erro(erro: Exception) -> str:
    texto = str(erro)
    if "429" in texto or "RESOURCE_EXHAUSTED" in texto:
        return "O limite gratuito do Gemini foi atingido. Espere um minuto e tente de novo."
    return f"Não consegui falar com o Gemini ({type(erro).__name__}). Confira a GEMINI_API_KEY no .env."


def ler_pdf(arquivo, nome: str) -> list[Document]:
    """Uma Document por página com texto, guardando arquivo e número da página."""
    paginas = []
    for i, pagina in enumerate(PdfReader(arquivo).pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            paginas.append(Document(page_content=texto, metadata={"arquivo": nome, "pagina": i}))
    return paginas


class IndiceFaiss:
    """Índice vetorial FAISS em memória: similaridade de cosseno (produto interno de vetores normalizados)."""

    def __init__(self, trechos: list[Document], embeddings):
        self.trechos, self.embeddings = trechos, embeddings
        vetores = self._normalizar(embeddings.embed_documents([t.page_content for t in trechos]))
        self.indice = faiss.IndexFlatIP(vetores.shape[1])
        self.indice.add(vetores)

    @staticmethod
    def _normalizar(vetores) -> np.ndarray:
        v = np.asarray(vetores, dtype="float32")
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def similarity_search(self, pergunta: str, k: int = 4) -> list[Document]:
        consulta = self._normalizar([self.embeddings.embed_query(pergunta)])
        _, posicoes = self.indice.search(consulta, min(k, len(self.trechos)))
        return [self.trechos[i] for i in posicoes[0] if i >= 0]


def criar_indice(paginas: list[Document], embeddings) -> IndiceFaiss:
    if not paginas:
        raise ValueError("Nenhum texto encontrado. O PDF pode ser escaneado (imagem); este app não faz OCR.")
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return IndiceFaiss(divisor.split_documents(paginas), embeddings)


def impressao_digital(arquivos: list[tuple[str, bytes]]) -> str:
    """Identifica o conjunto de arquivos, para não recriar o índice a cada interação."""
    h = hashlib.sha256()
    for nome, conteudo in sorted(arquivos):
        h.update(nome.encode())
        h.update(hashlib.sha256(conteudo).digest())
    return h.hexdigest()


def _formatar(trechos: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['arquivo']}, p. {d.metadata['pagina']}]\n{d.page_content}" for d in trechos)


def responder(pergunta: str, indice: IndiceFaiss, llm, historico: list[tuple[str, str]] | None = None) -> Resposta:
    """Busca os trechos mais parecidos e pede ao LLM uma resposta citando as páginas.

    `historico` são as últimas trocas (papel, texto), para perguntas de acompanhamento.
    """
    trechos = indice.similarity_search(pergunta, k=TRECHOS_POR_PERGUNTA)
    mensagens = PROMPT.invoke({
        "contexto": _formatar(trechos),
        "historico": (historico or [])[-6:],
        "pergunta": pergunta,
    })
    resposta = llm.invoke(mensagens)
    texto = resposta.content if isinstance(resposta.content, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in resposta.content)
    return Resposta(texto=texto.strip(), fontes=trechos)
