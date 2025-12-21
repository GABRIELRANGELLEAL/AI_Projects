import streamlit as st
import json
from datetime import date


class ChatTab:
    def __init__(self, client, meta_conector):
        """
        Classe para encapsular a aba de chat sobre dados/relatório.
        Permite reutilização deste bloco em diferentes apps Streamlit.
        """
        self.client = client
        self.meta_conector = meta_conector

    def create_initial_summary(self):
        st.subheader("Report and insights")
        
        # --- 1. Seleção de Período ---
        # Inicializa variáveis de estado
        if "show_dates" not in st.session_state:
            st.session_state.show_dates = False
        if "date_range" not in st.session_state:
            st.session_state.date_range = {"start": None , "end": None}

        # Botão para mostrar os inputs de data
        if st.button("Selecionar período"):
            st.session_state.show_dates = True # Ativa o flag para renderizar os inputs

        # Renderiza os inputs de data se o flag estiver ativo
        start, end = None, None
        if st.session_state.show_dates:
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Data inicial", value=date.today())
            with col2:
                end = st.date_input("Data final", value=date.today())

            # Salva as datas escolhidas no estado global
            st.session_state.date_range["start"] = start
            st.session_state.date_range["end"] = end
            
        # Formata as datas para o formato ISO (se existirem)
        if st.session_state.date_range["start"] and st.session_state.date_range["end"]:
            start = st.session_state.date_range['start']
            end = st.session_state.date_range['end']
            start_str = start.isoformat()
            end_str   = end.isoformat()
            
            # Mostra o período selecionado para o usuário
            st.write(
                f"Você selecionou: "
                f"{start_str} → {end_str}"
            )
            
        # --- 2. Geração do Relatório ---
        if st.button("Gerar relatório"):
            # Garante que as datas estejam disponíveis antes de prosseguir
            if not (start and end):
                st.error("Selecione um período válido.")
                return

            # Busca o ID do Vector Store (para futura integração RAG, se aplicável)
            vs_id = st.session_state.get("vs_id") 
            
            # Define os campos (métricas) a serem extraídos
            fields = ["date_start","campaign_name","adset_name","ad_name","spend","impressions","reach","inline_link_clicks"]
            
            # Extrai os dados de insights do meta_conector Ads
            dados = self.meta_conector.get_insights(fields = fields, since = f'{start_str}', until=f'{end_str}',time_increment = 'all_days')
            
            if dados:
                try:
                    # --- Chamada à API da OpenAI para gerar o Report ---
                    resp = self.client.responses.create(
                        model="gpt-4.1-mini",
                        input=[
                            {
                                "role": "developer",
                                # Instruções detalhadas (Prompt de Sistema) para o analista sênior
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
                                    # Passa os dados extraídos como JSON no input do usuário
                                    {"type": "input_text", "text": json.dumps(dados, ensure_ascii=False)}
                                ]
                            }
                        ],
                    )
                    
                    # Exibe o relatório gerado
                    st.subheader("Relatório")
                    st.markdown(resp.output_text)

                    # Ativa o modo chat e armazena os dados/relatório na sessão
                    st.session_state["chat_mode"] = True
                    st.session_state["dados"] = dados
                    st.session_state["relatorio"] = resp.output_text

                    # Inicializa histórico de conversa
                    if "messages" not in st.session_state:
                        st.session_state["messages"] = []

                except Exception as e:
                        st.error(f"Erro ao gerar relatório: {e}")
                        
    # --- 3. Chat sobre os Dados/Relatório ---
    def create_chat(self):
        if st.session_state.get("chat_mode", False):
            st.subheader("Chat sobre os dados e relatório")

            # --- 3a) CSS para fixar o input no rodapé (melhora usabilidade) ---
            st.markdown("""
                <style>
                    /* Fixa a barra de input no rodapé */
                    .stChatInputContainer { /* ... CSS rules ... */ }
                    /* Adiciona padding para o input não cobrir o conteúdo */
                    .block-container { /* ... CSS rules ... */ }
                </style>
            """, unsafe_allow_html=True)
            
            # --- 3b) Área do histórico (exibe mensagens em ordem reversa) ---
            chat_area = st.container()
            with chat_area:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # --- 3c) Input do Chat ---
            prompt = st.chat_input("Digite sua mensagem...")

            # --- 3d) Processa a Mensagem do Usuário ---
            if prompt:
                # Adiciona a mensagem do usuário ao histórico (no topo/início)
                st.session_state.messages.insert(0, {"role": "user", "content": prompt})
                
                # Monta o contexto para o GPT: System Prompt + Dados + Histórico
                contexto = [
                    {"role": "system", "content": "Você é um analista de marketing de dados que responde sobre campanhas.Sempre responda em um texto estruturado em no máximo dois parágrafos."},
                    # Inclui os dados extraídos para que o modelo possa referenciá-los
                    {"role": "assistant", "content": f"Dados disponíveis: {json.dumps(st.session_state['dados'], ensure_ascii=False)}"},
                ] + st.session_state["messages"] # Concatena o histórico

                # Chama a API da OpenAI para obter a resposta
                resp_chat = self.client.responses.create(
                    model="gpt-4.1-mini",
                    input=contexto
                )

                resp = resp_chat.output_text

                # Adiciona a resposta do assistente ao histórico (também no topo)
                st.session_state.messages.insert(0, {"role": "assistant", "content": resp})

                # Força o Streamlit a redesenhar a página para atualizar o histórico/input
                st.rerun()