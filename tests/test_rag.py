import io
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import criar_indice, impressao_digital, ler_pdf, responder  # noqa: E402


def _pdf(paginas: list[str]) -> bytes:
    """PDF mínimo com uma linha de texto por página (sem dependências extras)."""
    objetos, kids = [], []
    objetos.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objetos.append(None)  # páginas, preenchido depois
    objetos.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for texto in paginas:
        conteudo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
        objetos.append(b"<< /Length %d >>\nstream\n" % len(conteudo) + conteudo + b"\nendstream")
        objetos.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                       b"/Resources << /Font << /F1 3 0 R >> >> >>" % len(objetos))
        kids.append(len(objetos))
    objetos[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{k} 0 R" for k in kids).encode(), len(kids))
    saida, offsets = io.BytesIO(b"%PDF-1.4\n"), []
    saida.seek(0, 2)
    for i, o in enumerate(objetos, start=1):
        offsets.append(saida.tell())
        saida.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    xref = saida.tell()
    saida.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1))
    for off in offsets:
        saida.write(b"%010d 00000 n \n" % off)
    saida.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objetos) + 1, xref))
    return saida.getvalue()


EMB = DeterministicFakeEmbedding(size=64)


def test_le_pdf_guardando_arquivo_e_pagina():
    paginas = ler_pdf(io.BytesIO(_pdf(["Introducao ao projeto", "Resultados do teste"])), "relatorio.pdf")
    assert [p.metadata for p in paginas] == [{"arquivo": "relatorio.pdf", "pagina": 1},
                                             {"arquivo": "relatorio.pdf", "pagina": 2}]
    assert "Resultados" in paginas[1].page_content


def test_pdf_sem_texto_da_erro_claro():
    with pytest.raises(ValueError, match="escaneado"):
        criar_indice([], EMB)


def test_resposta_usa_trechos_e_historico():
    paginas = ler_pdf(io.BytesIO(_pdf(["O prazo de entrega e de 30 dias", "O preco e 100 reais"])), "contrato.pdf")
    indice = criar_indice(paginas, EMB)
    llm = FakeListChatModel(responses=["O prazo é de 30 dias [contrato.pdf, p. 1]."])
    r = responder("Qual o prazo?", indice, llm, historico=[("human", "oi"), ("ai", "olá")])
    assert r.texto.startswith("O prazo") and len(r.fontes) == 2
    assert {d.metadata["arquivo"] for d in r.fontes} == {"contrato.pdf"}


def test_prompt_inclui_contexto_com_paginas():
    from utils import PROMPT, _formatar

    paginas = ler_pdf(io.BytesIO(_pdf(["Texto A"])), "a.pdf")
    msgs = PROMPT.invoke({"contexto": _formatar(paginas), "historico": [], "pergunta": "?"}).to_messages()
    assert "[a.pdf, p. 1]" in msgs[0].content and "SOMENTE" in msgs[0].content


def test_impressao_digital_ignora_ordem_e_detecta_mudanca():
    a, b = ("a.pdf", b"1"), ("b.pdf", b"2")
    assert impressao_digital([a, b]) == impressao_digital([b, a])
    assert impressao_digital([a]) != impressao_digital([("a.pdf", b"outro")])
