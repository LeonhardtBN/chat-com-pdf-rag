"""RAG sobre PDFs: extração por página, divisão em trechos, índice FAISS e resposta com fontes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
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

    return ChatGoogleGenerativeAI(model=MODELO_CHAT), GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)


def ler_pdf(arquivo, nome: str) -> list[Document]:
    """Uma Document por página com texto, guardando arquivo e número da página."""
    paginas = []
    for i, pagina in enumerate(PdfReader(arquivo).pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            paginas.append(Document(page_content=texto, metadata={"arquivo": nome, "pagina": i}))
    return paginas


def criar_indice(paginas: list[Document], embeddings) -> FAISS:
    if not paginas:
        raise ValueError("Nenhum texto encontrado. O PDF pode ser escaneado (imagem); este app não faz OCR.")
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return FAISS.from_documents(divisor.split_documents(paginas), embeddings)


def impressao_digital(arquivos: list[tuple[str, bytes]]) -> str:
    """Identifica o conjunto de arquivos, para não recriar o índice a cada interação."""
    h = hashlib.sha256()
    for nome, conteudo in sorted(arquivos):
        h.update(nome.encode())
        h.update(hashlib.sha256(conteudo).digest())
    return h.hexdigest()


def _formatar(trechos: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['arquivo']}, p. {d.metadata['pagina']}]\n{d.page_content}" for d in trechos)


def responder(pergunta: str, indice: FAISS, llm, historico: list[tuple[str, str]] | None = None) -> Resposta:
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
