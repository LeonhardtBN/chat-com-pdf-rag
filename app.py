"""Chat com PDF (RAG). Rode com: streamlit run app.py"""

import io
import os

import streamlit as st
from dotenv import load_dotenv

from utils import criar_indice, impressao_digital, ler_pdf, modelos_padrao, responder

load_dotenv()
st.set_page_config(page_title="Converse com seus PDFs", page_icon=":books:")
st.title("Converse com seus PDFs")
st.caption("Envie um ou mais PDFs e faça perguntas. As respostas citam o arquivo e a página de onde vieram.")

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.error("Defina GEMINI_API_KEY no arquivo .env (veja .env.example). "
             "A chave é gratuita: https://aistudio.google.com/apikey")
    st.stop()


@st.cache_resource
def carregar_modelos():
    return modelos_padrao()


llm, embeddings = carregar_modelos()

with st.sidebar:
    st.subheader("Seus arquivos")
    enviados = st.file_uploader("Envie PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Limpar conversa", width="stretch"):
        st.session_state.mensagens = []

if not enviados:
    st.info("Envie um PDF na barra lateral para começar.")
    st.stop()

# O índice só é recriado quando o conjunto de arquivos muda (antes era recriado a cada pergunta).
arquivos = [(f.name, f.getvalue()) for f in enviados]
chave = impressao_digital(arquivos)
if st.session_state.get("chave_indice") != chave:
    with st.spinner("Lendo os PDFs e criando a base de conhecimento..."):
        paginas = [p for nome, dados in arquivos for p in ler_pdf(io.BytesIO(dados), nome)]
        try:
            st.session_state.indice = criar_indice(paginas, embeddings)
        except ValueError as e:
            st.error(str(e))
            st.stop()
    st.session_state.chave_indice = chave
    st.session_state.mensagens = []
    st.sidebar.success(f"{len(paginas)} páginas indexadas.")

st.session_state.setdefault("mensagens", [])
for m in st.session_state.mensagens:
    with st.chat_message(m["papel"]):
        st.markdown(m["texto"])

if pergunta := st.chat_input("Faça sua pergunta sobre os documentos"):
    with st.chat_message("user"):
        st.markdown(pergunta)
    historico = [("human" if m["papel"] == "user" else "ai", m["texto"]) for m in st.session_state.mensagens]
    with st.chat_message("assistant"):
        with st.spinner("Procurando nos documentos..."):
            resposta = responder(pergunta, st.session_state.indice, llm, historico)
        st.markdown(resposta.texto)
        with st.expander("Trechos usados"):
            for d in resposta.fontes:
                st.markdown(f"**{d.metadata['arquivo']}, p. {d.metadata['pagina']}**")
                st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
    st.session_state.mensagens += [{"papel": "user", "texto": pergunta},
                                   {"papel": "assistant", "texto": resposta.texto}]
