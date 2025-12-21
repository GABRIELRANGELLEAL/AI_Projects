import pandas as pd
import streamlit as st
import json
import io
import openai
import tabulate
import requests
import os
from datetime import date
from dotenv import load_dotenv, find_dotenv
from  meta_adds_connect import insights_meta as im
import urllib.parse

_ = load_dotenv(find_dotenv(), override=True)

APP_ID = os.getenv("META_APP_ID")
APP_SECRET = os.getenv("META_APP_SECRET")
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
 

# =========================================================
# ========== Helpers =========
# =========================================================
def format_df_for_markdown(df: pd.DataFrame, max_rows=200, max_cols=30, float_round=4) -> pd.DataFrame:
    """
    - Limita linhas/colunas para não gerar textos enormes
    - Arredonda floats para reduzir "ruído" de dígitos
    """
    # limita colunas
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols].copy()

    # limita linhas
    if len(df) > max_rows:
        df = df.head(max_rows).copy()

    # arredonda floats
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].round(float_round)

    return df

def df_to_md_bytesio(df: pd.DataFrame, file_basename: str) -> io.BytesIO:
    """
    Converte um DataFrame em Markdown (tabela) e retorna como BytesIO ".md".
    """
    df_md = format_df_for_markdown(df)
    md_text = df_md.to_markdown(index=False)
    # título e dica de truncamento
    md_full = f'''
        # {file_basename}
        _Tabela convertida automaticamente para Markdown._
        {md_text}
    '''
    bio = io.BytesIO(md_full.encode("utf-8"))
    bio.name = f"{file_basename}.md"   # extensão suportada pelo file_search
    bio.seek(0)
    return bio

def csv_to_md_bytesio(uploaded) -> io.BytesIO:
    uploaded.seek(0)
    df = pd.read_csv(uploaded)  # ajuste sep=... se necessário
    base = uploaded.name.rsplit(".", 1)[0]
    return df_to_md_bytesio(df, base)

def excel_to_md_bytesio(uploaded) -> io.BytesIO:
    uploaded.seek(0)
    # lê a primeira sheet por padrão; se quiser, liste as sheets e concatene
    df = pd.read_excel(uploaded)
    base = uploaded.name.rsplit(".", 1)[0]
    return df_to_md_bytesio(df, base)

def to_bytesio_with_name(uploaded) -> io.BytesIO:
    """
    Para PDFs já suportados nativamente.
    """
    data = uploaded.read()
    bio = io.BytesIO(data)
    bio.name = uploaded.name
    bio.seek(0)
    return bio

# =========================================================
# ========== Aba Input your data =========
# =========================================================

def import_data():
    # ========== upload files =========
    st.subheader("Upload arquivo")
    vs = client.vector_stores.create(name="docs_sessao_local")
    #st.write("Vector Store criado:", vs.id)
    st.session_state["vs_id"] = vs.id  # só guarda o ID

    uploaded_files = st.file_uploader(
        "Escolha arquivos (PDF, CSV, Excel, CSV)",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Arquivos recebidos: {[uf.name for uf in uploaded_files]}")
        success, failed = 0, 0

        for uf in uploaded_files:
            name_lower = uf.name.lower()

            if name_lower.endswith(".pdf"):
                fs = to_bytesio_with_name(uf)
            elif name_lower.endswith(".csv"):
                fs = csv_to_md_bytesio(uf)
            elif name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
                fs = excel_to_md_bytesio(uf)
            else:
                st.warning(f"Pulado: {uf.name} (tipo não suportado)")
                continue

            try:
                batch = client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vs.id,
                    files=[fs]
                )
                st.write(f"✅ Indexado: {fs.name} | status: {getattr(batch, 'status', 'desconhecido')}")
                success += 1
            except Exception as e:
                st.error(f"❌ Falha ao indexar {fs.name}: {e}")
                failed += 1
            finally:
                try:
                    fs.close()
                except:
                    pass
        st.write(f"Resumo: Sucesso={success} | Falhas={failed}")
    
    # ========== FACE ADDS CONECTOR =========
    st.subheader("Conexão direta")
    # st.title("🔗 Conectar ao Facebook Ads")
    REDIRECT_URI = "http://localhost:8501" # ajuste para sua URL pública quando em prod
    # Monta a URL de login (OAuth2)
    auth_url = (
        f"https://www.facebook.com/v19.0/dialog/oauth?"
        f"client_id={APP_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}&scope=ads_read"
    )

    # Se ainda não tem token → mostra botão de login
    if "fb_token" not in st.session_state:
        auth_url = "https://www.facebook.com/v19.0/dialog/oauth?client_id=SEU_CLIENT_ID&redirect_uri=SUA_REDIRECT_URI&scope=ads_read"
        st.markdown(
            f"""
            <style>
            .fb-button {{
                display: inline-block;
                padding: 12px 24px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: #1877F2; /* Azul do Facebook */
                border: none;
                border-radius: 8px;
                text-align: center;
                text-decoration: none;
                transition: background-color 0.3s;
            }}
            .fb-button:hover {{
                background-color: #145dbf;
            }}
            </style>

            <a href="{auth_url}" target="_blank" class="fb-button">
                Conectar ao Facebook Ads
            </a>
            """,
            unsafe_allow_html=True
        )

        # Captura o "code" que o Facebook devolve após login
        params = st.query_params
        if "code" in params:
            code = params["code"][0]

            # Troca o code por access_token
            token_url = (
                f"https://graph.facebook.com/v19.0/oauth/access_token?"
                f"client_id={APP_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
                f"&client_secret={APP_SECRET}&code={code}"
            )
            resp = requests.get(token_url).json()
            if "access_token" in resp:
                st.session_state["fb_token"] = resp["access_token"]
                st.success("✅ Conectado com sucesso!")
            else:
                st.error(f"Erro ao obter token: {resp}")
    else:
        st.success("✅ Já conectado ao Facebook Ads")
        token = st.session_state["fb_token"]

# =========================================================
# ========== Aba Report e Insights =========
# =========================================================
def create_report():
    st.subheader("Report and insights")
    # ---------------------------
    # BOTÃO QUE ATIVA AS DATAS
    # ---------------------------
    # Usamos st.session_state para guardar variáveis entre "reruns" do Streamlit.
    # show_dates: controla se as caixas de data devem aparecer
    # date_range: dicionário para armazenar as duas datas escolhidas
    if "show_dates" not in st.session_state:
        st.session_state.show_dates = False
    if "date_range" not in st.session_state:
        st.session_state.date_range = {"start": None , "end": None}

    # Quando o botão é clicado, o Streamlit dá um "rerun".
    # Se o clique for detectado (True), ativamos o flag para mostrar os inputs.
    if st.button("Selecionar período"):
        st.session_state.show_dates = True

    # Aqui renderizamos duas caixas de data lado a lado (colunas).
    # Também já salvamos os valores escolhidos no session_state.
    # start, end = date.today(), date.today() 
    if st.session_state.show_dates:
        col1, col2 = st.columns(2)
        with col1:
            # date_input retorna um objeto date
            start = st.date_input("Data inicial", value=date.today())
        with col2:
            end = st.date_input("Data final", value=date.today())

        # Salva no estado global para reutilizar em qualquer parte do app
        st.session_state.date_range["start"] = start
        st.session_state.date_range["end"] = end
    if start and end:
        start_str = start.isoformat()
        end_str   = end.isoformat()
    # mostrando range date pro usário
    # Basta acessar st.session_state.date_range["start"] e ["end"] onde precisar.
    if st.session_state.date_range["start"] and st.session_state.date_range["end"]:
        st.write(
            f"Você selecionou: "
            f"{st.session_state.date_range['start']} → {st.session_state.date_range['end']}"
        )
    # ----------------------------------------
    # extrainddo dados da api e fazendo  relatório
    # ----------------------------------------
    if st.button("Gerar relatório"):
        # valida se temos um vector store disponível
        vs_id = st.session_state.get("vs_id")
        meta = im(APP_ID, APP_SECRET, ACCESS_TOKEN, AD_ACCOUNT_ID)
        fields = ["date_start","campaign_name","adset_name","ad_name","spend","impressions","reach","inline_link_clicks"]
        dados = meta.get_insights(fields = fields, since = f'{start_str}', until=f'{end_str}',time_increment = 'all_days')
        if dados:
            try:
                resp = client.responses.create(
                    model="gpt-4.1-mini",  # modelo de texto compatível com tools
                    input=[
                        {
                            "role": "developer",
                            "content": f'''
                                Você receberá no user input uma lista de dicionários contendo resultados de campanhas do cliente. 
                                Atue como um analista de marketing de dados sênior e produza um report estruturado no seguinte formato: 
                                📊 Reporte de Performance – Campanhas 
                                    1. Resumo Executivo 
                                        - 🔥 Destaque os **melhores desempenhos** (com métricas como CTR, CPC, CPM, cliques etc (coloque as métricas em bullet points)) 
                                        - ❌ Destaque os **piores desempenhos** (com métricas como CTR, CPC, CPM, cliques etc (coloque as métricas em bullet points)) 
                                    2. Recomendações 
                                        - 💰 Sugestões sobre **alocação de budget** (bullet points) 
                                        - 🧪 Demais sugestões
                                Regras: 
                                    - Use métricas por criativo/adset/campanha quando fizer sentido. 
                                    - cuidado quando for usar R$, não coloque o R$ coloque apenas valores
                            '''
                        },
                        {
                            "role": "user",
                            "content": [
                                # dados como JSON em texto:
                                {"type": "input_text", "text": json.dumps(dados, ensure_ascii=False)}
                            ]
                        }
                    ],
                )
                st.subheader("Relatório")
                st.markdown(resp.output_text)

                # Ativa chat
                st.session_state["chat_mode"] = True
                st.session_state["dados"] = dados
                st.session_state["relatorio"] = resp.output_text

                # Inicializa histórico de conversa vazio (só para registrar iterações do chat)
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []

            except Exception as e:
                    st.error(f"Erro ao gerar relatório: {e}")
                    
    # # --- Parte 2: Chat ---
    if st.session_state.get("chat_mode", False):
        st.subheader("Chat sobre os dados e relatório")

        # --- 1) CSS: fixa o input no rodapé e evita sobreposição ---
        st.markdown("""
            <style>
                /* Fixa a barra de input no rodapé da janela */
                .stChatInputContainer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                z-index: 999; /* fica por cima de tudo */
                }
                /* Dá espaço extra no fim da página para o input não cobrir as mensagens */
                .block-container {
                padding-bottom: 7rem; /* ajuste fino se quiser */
                }
            </style>
        """, unsafe_allow_html=True)
        
        # --- 3) Área do histórico (mais novas em cima) ---
        chat_area = st.container()
        with chat_area:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # --- 4) Input SEMPRE no fim do script (mas fixado embaixo pelo CSS) ---
        prompt = st.chat_input("Digite sua mensagem...")

        # --- 5) Processa envio: insere no TOPO e reroda a app ---
        if prompt:
            # Mensagem do usuário no topo
            st.session_state.messages.insert(0, {"role": "user", "content": prompt})
            
            # Monta contexto para API (dados + relatório + histórico do chat)
            contexto = [
                {"role": "system", "content": "Você é um analista de marketing de dados que responde sobre campanhas.Sempre responda em um texto estruturado em no máximo dois parágrafos."},
                {"role": "assistant", "content": f"Dados disponíveis: {json.dumps(st.session_state['dados'], ensure_ascii=False)}"},
            ] + st.session_state["messages"]

            resp_chat = client.responses.create(
                model="gpt-4.1-mini",
                input=contexto
            )

            resp = resp_chat.output_text

            # Resposta do assistente também no topo (acima da do user)
            st.session_state.messages.insert(0, {"role": "assistant", "content": resp})

            # Importante: reroda para redesenhar com o input lá embaixo e histórico atua lizado
            st.rerun()

# =========================================================
# ========== Run App =========
# =========================================================

def main():
    st.header('Mkt Analytics Tool')
    tab1, tab2 = st.tabs(["Input your data", "Report and insights"])
    with tab1:
        import_data()
    with tab2:
        create_report()

if __name__ == '__main__':
    main()
