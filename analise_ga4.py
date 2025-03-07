import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
from io import StringIO
import base64

# Verificar dependências opcionais
optional_deps_available = {}

try:
    import seaborn as sns
    optional_deps_available['seaborn'] = True
except ImportError:
    optional_deps_available['seaborn'] = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    optional_deps_available['scikit-learn'] = True
except ImportError:
    optional_deps_available['scikit-learn'] = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    optional_deps_available['statsmodels'] = True
except ImportError:
    optional_deps_available['statsmodels'] = False

try:
    import networkx as nx
    optional_deps_available['networkx'] = True
except ImportError:
    optional_deps_available['networkx'] = False

try:
    from wordcloud import WordCloud
    optional_deps_available['wordcloud'] = True
except ImportError:
    optional_deps_available['wordcloud'] = False

# Configuração da página e cache
st.set_page_config(
    page_title="Analytics Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar cache para melhorar o desempenho
@st.cache_data(ttl=3600)
def load_cached_data(file_content):
    return pd.read_csv(StringIO(file_content))

# Estilo CSS personalizado atualizado para compatibilidade com tema escuro
st.markdown("""
<style>
    /* Estilos gerais adaptáveis ao tema */
    .main {
        background-color: transparent;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    
    /* Cartões e destaques com cores adaptáveis */
    .highlight {
        background-color: rgba(100, 150, 250, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(100, 150, 250, 0.2);
    }
    .highlight-success {
        background-color: rgba(70, 200, 120, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(70, 200, 120, 0.2);
    }
    .highlight-warning {
        background-color: rgba(250, 180, 70, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(250, 180, 70, 0.2);
    }
    .highlight-metrics {
        background-color: rgba(120, 120, 250, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        border: 1px solid rgba(120, 120, 250, 0.2);
    }
    
    /* Cartões de métricas */
    .metric-card {
        background-color: rgba(30, 30, 30, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        text-align: center;
        width: 24%;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(120, 120, 250, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Estilos para status de login */
    .logged-in {
        background-color: rgba(70, 200, 120, 0.1);
        border: 1px solid rgba(70, 200, 120, 0.2);
    }
    .not-logged-in {
        background-color: rgba(120, 120, 120, 0.1);
        border: 1px solid rgba(120, 120, 120, 0.2);
    }
    
    /* Estilos de tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(120, 120, 120, 0.1);
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 0px 16px;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Botão de download */
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white !important;
        text-decoration: none;
        border-radius: 4px;
        margin-top: 0.5rem;
        text-align: center;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    
    /* Tabelas e data frames */
    .dataframe {
        border-collapse: collapse;
    }
    .dataframe th {
        background-color: rgba(120, 120, 120, 0.1);
    }
    .dataframe td, .dataframe th {
        border: 1px solid rgba(120, 120, 120, 0.2);
        padding: 8px;
    }
    
    /* Cards informativos */
    .info-card {
        background-color: rgba(70, 130, 180, 0.1);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(70, 130, 180, 0.2);
    }
    
    .info-card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .feature-card-one {
    background-color: rgba(100, 200, 150, 0.1); /* Tom verde-água suave */
    border-radius: 8px;
    padding: 20px;
    border: 1px solid rgba(100, 200, 150, 0.2);
    transition: transform 0.3s ease;
    }

    .feature-card-one:hover {
        transform: translateY(-5px);
    }

    .feature-card-two {
        background-color: rgba(200, 150, 100, 0.1); /* Tom laranja-terroso suave */
        border-radius: 8px;
        padding: 20px;
        border: 1px solid rgba(200, 150, 100, 0.2);
        transition: transform 0.3s ease;
    }

    .feature-card-two:hover {
        transform: translateY(-5px);
    }

    .feature-card-three {
        background-color: rgba(150, 100, 200, 0.1); /* Tom roxo suave */
        border-radius: 8px;
        padding: 20px;
        border: 1px solid rgba(150, 100, 200, 0.2);
        transition: transform 0.3s ease;
    }

    .feature-card-three:hover {
        transform: translateY(-5px);
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Ajustar texto para temas escuros */
    h1, h2, h3, h4, h5, p, li, td, th, div {
        color: inherit;
    }
    
    /* Responsividade para mobile */
    @media (max-width: 768px) {
        .metric-card {
            width: 48%;
        }
        .highlight-metrics {
            flex-wrap: wrap;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Função para converter timestamp para formato legível
def format_timestamp(timestamp):
    return datetime.fromtimestamp(int(timestamp)/1000000).strftime('%d/%m/%Y %H:%M:%S')

# Função para extrair apenas a data do timestamp
def extract_date(timestamp):
    return datetime.fromtimestamp(int(timestamp)/1000000).date()

# Função para extrair hora do dia
def extract_hour(timestamp):
    return datetime.fromtimestamp(int(timestamp)/1000000).hour

# Função para calcular dia da semana
def extract_weekday(timestamp):
    return datetime.fromtimestamp(int(timestamp)/1000000).strftime('%A')

# Função para carregar os dados
def load_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            # Ler conteúdo do arquivo e usar cache
            file_content = uploaded_file.getvalue().decode('utf-8')
            df = load_cached_data(file_content)
        elif file_path is not None and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                file_content = f.read()
            df = load_cached_data(file_content)
        else:
            st.error("Arquivo não encontrado. Por favor, verifique o caminho ou faça upload do arquivo.")
            return None
        
        # Validar se o arquivo tem o formato esperado
        colunas_esperadas = ['event_name', 'user_id', 'client_id', 'session_number', 'session_id', 
                            'session_campaign_id', 'session_campaign_name', 'session_source', 
                            'session_medium', 'event_timestamp']
        colunas_ausentes = [col for col in colunas_esperadas if col not in df.columns]
        
        if colunas_ausentes:
            st.error(f"O arquivo CSV não contém todas as colunas necessárias. Faltam: {', '.join(colunas_ausentes)}")
            st.warning("Por favor, verifique se você está usando a query SQL correta para exportar os dados do GA4 conforme instruções na página inicial.")
            return None
        
        # Converter as colunas para os tipos corretos
        df['session_number'] = df['session_number'].astype(int)
        df['event_timestamp'] = df['event_timestamp'].astype(float)
        
        # Adicionar coluna para identificar se o usuário está logado
        df['is_logged_in'] = df['user_id'].notna()
        
        # Adicionar coluna com timestamp formatado e data
        df['formatted_timestamp'] = df['event_timestamp'].apply(format_timestamp)
        df['date'] = df['event_timestamp'].apply(extract_date)
        df['hour'] = df['event_timestamp'].apply(extract_hour)
        df['weekday'] = df['event_timestamp'].apply(extract_weekday)
        
        # Adicionar coluna para tempo desde primeiro evento (para análise de funil)
        df['time_since_first_event'] = df.groupby(['client_id', 'session_number'])['event_timestamp'].transform(
            lambda x: (x - x.min()) / 1000000  # Converter para segundos
        )
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None

# Função para exportar dados como CSV
def get_csv_download_link(df, filename="dados_exportados.csv", button_text="Baixar dados como CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'
    return href

# Função para highlight login (compatível com tema escuro)
def highlight_login(row):
    if row.get('Usuário Logado', False):
        return ['background-color: rgba(70, 200, 120, 0.1)'] * len(row)
    else:
        return ['background-color: rgba(120, 120, 120, 0.1)'] * len(row)

# Função para criar um funil de eventos
def create_funnel_chart(df, event_sequence, title="Funil de Conversão"):
    # Contar quantos usuários realizaram cada evento
    event_counts = {}
    remaining_users = set(df['client_id'].unique())
    
    for event in event_sequence:
        users_with_event = set(df[df['event_name'] == event]['client_id'].unique())
        users_in_funnel = remaining_users.intersection(users_with_event)
        event_counts[event] = len(users_in_funnel)
        remaining_users = users_in_funnel
    
    # Criar dataframe para o gráfico
    funnel_df = pd.DataFrame({
        'Evento': list(event_counts.keys()),
        'Usuários': list(event_counts.values())
    })
    
    # Adicionar taxa de conversão
    funnel_df['Taxa de Conversão'] = 100.0
    for i in range(1, len(funnel_df)):
        prev_users = funnel_df.iloc[i-1]['Usuários']
        curr_users = funnel_df.iloc[i]['Usuários']
        funnel_df.loc[i, 'Taxa de Conversão'] = round((curr_users / prev_users * 100 if prev_users > 0 else 0), 1)
    
    # Criar gráfico
    fig = go.Figure(go.Funnel(
        y=funnel_df['Evento'],
        x=funnel_df['Usuários'],
        textposition="inside",
        textinfo="value+percent initial",
        texttemplate="%{value} (%{percentInitial:.1%})",
        marker=dict(color=px.colors.sequential.Blues)
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig, funnel_df

# Função para calcular métricas de sessão
def calculate_session_metrics(df):
    # Agrupar por sessão e calcular métricas
    session_metrics = df.groupby(['client_id', 'session_number']).agg({
        'event_timestamp': ['min', 'max', 'count'],
        'is_logged_in': 'any'
    }).reset_index()
    
    # Renomear colunas
    session_metrics.columns = ['client_id', 'session_number', 'start_time', 'end_time', 'events_count', 'is_logged_in']
    
    # Calcular duração da sessão em segundos
    session_metrics['duration_seconds'] = (session_metrics['end_time'] - session_metrics['start_time']) / 1000000
    
    return session_metrics

# Página inicial com explicação da ferramenta
def show_home_page():
    st.title("🔍 Analytics Insights Dashboard")
    
    
    
    # Verificar dependências e mostrar instruções se necessário
    missing_deps = [dep for dep, available in optional_deps_available.items() if not available]
    if missing_deps:
        st.warning("### ⚠️ Algumas funcionalidades avançadas não estão disponíveis")
        st.markdown(f"""
        Para habilitar todas as funcionalidades do dashboard, por favor instale as seguintes bibliotecas:
        
        ```
        pip install {' '.join(missing_deps)}
        ```
        
        Depois de instalar, reinicie o aplicativo para acessar os recursos avançados.
        """)
    
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Bem-vindo ao Analytics Insights Dashboard</div>
        <p>Esta ferramenta foi projetada para transformar seus dados brutos do Google Analytics 4 (GA4) em insights acionáveis que impulsionam decisões estratégicas.</p>
        <p>Seja você um analista de marketing, gerente de produto ou executivo, nosso dashboard oferece visualizações intuitivas e análises profundas que revelam padrões ocultos no comportamento do usuário.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌟 Principais Funcionalidades")
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card-one">
            <div class="feature-title">👤 Análise por Usuário</div>
            <p>Examine o comportamento individual de usuários ao longo do tempo, identificando padrões de engajamento e oportunidades de personalização.</p>
            <p><strong>Benefícios:</strong> Compreensão profunda das jornadas individuais e identificação de usuários de alto valor.</p>
        </div>
         <div class="feature-card-two">
            <div class="feature-title">📊 Análise de Eventos</div>
            <p>Avalie como diferentes eventos de interação se comportam ao longo das sessões dos usuários, segmentando por tipo, temporalidade e origem do tráfego, com análises detalhadas de funil, distribuição horária e correlações entre eventos-chave.</p>
            <p><strong>Benefícios:</strong> Otimização de conversões por segmento, identificação de pontos de atrito no fluxo de usuário e descoberta de padrões sazonais que impactam o comportamento dos visitantes.</p>
        </div>
        <div class="feature-card-three">
            <div class="feature-title">📈 Jornada do Usuário</div>
            <p>Analise as interações dos usuários através de múltiplas perspectivas: caminhos de navegação frequentes, tempos de conversão entre eventos, funis personalizados e visualização de fluxos com análise de drop-off.</p>
            <p><strong>Benefícios:</strong> Identificação de gargalos de conversão, otimização dos caminhos de navegação, e compreensão detalhada do comportamento do usuário em cada etapa da jornada.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Como usar este dashboard")
    
    st.markdown("""
    <ol>
        <li><strong>Carregar dados:</strong> Comece fazendo upload do seu arquivo CSV de dados do GA4 ou especificando o caminho do arquivo no painel lateral.</li>
        <li><strong>Selecionar período:</strong> Use o filtro de data para focar em períodos específicos de interesse.</li>
        <li><strong>Explorar módulos:</strong> Navegue pelas diferentes páginas de análise usando o menu lateral para obter insights específicos.</li>
        <li><strong>Exportar resultados:</strong> Cada módulo oferece opções para exportar dados e visualizações para relatórios externos.</li>
    </ol>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Por que usar análise avançada de GA4?")
    
    st.markdown("""
    <div class="info-card">
        <p>Os dados do GA4 contêm informações valiosas que muitas vezes permanecem inexploradas nas interfaces padrão. Nossa ferramenta desbloqueia este valor através de:</p>
        <ul>
            <li><strong>Análise multi-sessão:</strong> Compreenda como os usuários se comportam ao longo de múltiplas visitas, não apenas em sessões isoladas.</li>
            <li><strong>Visualização de canais:</strong> Avalie como diferentes fontes de tráfego contribuem para suas conversões ao longo do tempo.</li>
            <li><strong>Insights acionáveis:</strong> Transforme dados brutos em estratégias concretas para otimização de produtos e marketing.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Informações sobre o formato CSV esperado
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Sobre o formato de dados esperado</div>
        <p>Este dashboard foi projetado para analisar dados do Google Analytics 4 (GA4) exportados do BigQuery.</p>
        <p>O CSV esperado é da tabela <code>events_</code> do BigQuery, que é criada após a vinculação do GA4 com o BigQuery.</p>
        <p>Para extrair os dados corretamente, utilize a seguinte query SQL no BigQuery:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
        SELECT
        event_name,
        user_id,
        user_pseudo_id AS client_id,
        (
            SELECT ep.value.int_value
            FROM UNNEST(event_params) AS ep
            WHERE ep.key = 'ga_session_number'
        ) AS session_number,
        (
            SELECT ep.value.int_value
            FROM UNNEST(event_params) AS ep
            WHERE ep.key = 'ga_session_id'
        ) AS session_id,

        session_traffic_source_last_click.manual_campaign.campaign_id   AS session_campaign_id,
        session_traffic_source_last_click.manual_campaign.campaign_name AS session_campaign_name,
        session_traffic_source_last_click.manual_campaign.source         AS session_source,
        session_traffic_source_last_click.manual_campaign.medium         AS session_medium,
        session_traffic_source_last_click.manual_campaign.term           AS session_term,
        session_traffic_source_last_click.manual_campaign.content        AS session_content,
        event_timestamp
        FROM `ID da tabela`
        ORDER BY event_timestamp ASC
        """, language="sql")
    
    st.markdown("### 🔧 Comece agora")
    st.markdown("Carregue seus dados usando o painel lateral para descobrir insights que podem transformar sua estratégia digital.")

# Página de análise por usuário
def show_user_analysis_page(filtered_df):
    st.header("👤 Análise por Usuário")
    
    # Opções de busca com duas opções
    search_option = st.radio(
        "Buscar por:",
        ["Client ID", "User ID"]
    )
    
    # Campo de busca adaptável ao tipo selecionado
    if search_option == "Client ID":
        client_id_input = st.text_input("Digite o Client ID:")
        if client_id_input:
            # Converter client_ids para string para garantir comparação consistente
            client_ids_list = [str(cid).strip() for cid in filtered_df['client_id'].unique()]
            client_id_input = str(client_id_input).strip()
            
            if client_id_input in client_ids_list:
                # Encontrar o client_id original no dataframe que corresponde ao input
                for cid in filtered_df['client_id'].unique():
                    if str(cid).strip() == client_id_input:
                        selected_id = cid
                        search_column = 'client_id'
                        break
            else:
                st.error(f"Client ID '{client_id_input}' não encontrado nos dados.")
                st.info("Tente selecionar da lista abaixo para ver os IDs disponíveis.")
                selected_id = None
        else:
            # Opção de selecionar da lista
            client_ids = filtered_df['client_id'].unique()
            selected_id = st.selectbox("Ou selecione um Client ID:", client_ids)
            search_column = 'client_id'
    
    else:  # User ID
        user_id_input = st.text_input("Digite o User ID:")
        if user_id_input:
            user_data = filtered_df[filtered_df['user_id'] == user_id_input]
            if not user_data.empty:
                # Verificar se esse user_id está associado a mais de um client_id
                client_ids_for_user = user_data['client_id'].unique()
                if len(client_ids_for_user) > 1:
                    st.warning(f"Este User ID está associado a {len(client_ids_for_user)} Client IDs diferentes!")
                    st.write("Client IDs associados:")
                    for cid in client_ids_for_user:
                        st.write(f"- {cid}")
                    selected_id = st.selectbox("Selecione um Client ID para análise:", client_ids_for_user)
                else:
                    selected_id = client_ids_for_user[0]
                    search_column = 'client_id'
            else:
                st.error(f"User ID '{user_id_input}' não encontrado nos dados.")
                selected_id = None
        else:
            # Opção de selecionar da lista de user_ids não vazios
            valid_user_ids = filtered_df[filtered_df['user_id'].notna()]['user_id'].unique()
            if len(valid_user_ids) > 0:
                selected_user_id = st.selectbox("Selecione um User ID:", valid_user_ids)
                user_data = filtered_df[filtered_df['user_id'] == selected_user_id]
                # Verificar se esse user_id está associado a mais de um client_id
                client_ids_for_user = user_data['client_id'].unique()
                if len(client_ids_for_user) > 1:
                    st.warning(f"Este User ID está associado a {len(client_ids_for_user)} Client IDs diferentes!")
                    st.write("Client IDs associados:")
                    for cid in client_ids_for_user:
                        st.write(f"- {cid}")
                    selected_id = st.selectbox("Selecione um Client ID para análise:", client_ids_for_user)
                else:
                    selected_id = client_ids_for_user[0]
                    search_column = 'client_id'
            else:
                st.warning("Não foram encontrados User IDs nos dados.")
                selected_id = None
    
    # Verificar se um client_id foi selecionado/encontrado
    if selected_id and search_column == 'client_id':
        # Verificar se este client_id tem múltiplos user_ids
        user_data = filtered_df[filtered_df['client_id'] == selected_id]
        user_ids = user_data[user_data['user_id'].notna()]['user_id'].unique()
        
        if len(user_ids) > 1:
            st.warning(f"Este Client ID ('{selected_id}') está associado a {len(user_ids)} User IDs diferentes!")
            st.write("User IDs associados:")
            for uid in user_ids:
                st.write(f"- {uid}")
        
        # Exibir resumo das sessões do usuário
        st.subheader(f"Sessões do cliente {selected_id}")
        
        # Métricas resumidas do usuário
        total_sessions = user_data['session_number'].nunique()
        total_events = len(user_data)
        first_seen = user_data['date'].min()
        last_seen = user_data['date'].max()
        days_active = (last_seen - first_seen).days + 1
        
        # Exibir métricas em cards (removida animação)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Total de Sessões**\n\n{total_sessions}")
        with col2:
            st.write(f"**Total de Eventos**\n\n{total_events}")
        with col3:
            st.write(f"**Primeira Visita**\n\n{first_seen.strftime('%d/%m/%Y')}")
        with col4:
            st.write(f"**Última Visita**\n\n{last_seen.strftime('%d/%m/%Y')}")
        
        # Agrupar as informações das sessões
        sessions = user_data.groupby('session_number').agg({
            'session_id': 'first',
            'session_source': 'first',
            'session_medium': 'first',
            'event_timestamp': ['min', 'max'],
            'is_logged_in': 'any',
            'user_id': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
        }).reset_index()
        
        # Reorganizar colunas
        sessions.columns = ['session_number', 'session_id', 'session_source', 'session_medium', 
                            'start_timestamp', 'end_timestamp', 'is_logged_in', 'user_id']
        
        sessions['início'] = sessions['start_timestamp'].apply(format_timestamp)
        sessions['fim'] = sessions['end_timestamp'].apply(format_timestamp)
        sessions['duração'] = (sessions['end_timestamp'] - sessions['start_timestamp']) / 1000000  # Em segundos
        
        # Criar dataframe de resumo de sessão
        session_summary = pd.DataFrame({
            'Nº da Sessão': sessions['session_number'],
            'ID da Sessão': sessions['session_id'],
            'Origem': sessions['session_source'],
            'Mídia': sessions['session_medium'],
            'Início': sessions['início'],
            'Duração (s)': sessions['duração'].round(1),
            'User ID': sessions['user_id'],
            'Usuário Logado': sessions['is_logged_in']
        })
        
        # Contar eventos por sessão
        events_per_session = user_data.groupby('session_number').size().reset_index(name='num_events')
        session_summary = pd.merge(session_summary, events_per_session, 
                                  left_on='Nº da Sessão', right_on='session_number')
        session_summary.rename(columns={'num_events': 'Nº de Eventos'}, inplace=True)
        session_summary.drop(columns=['session_number'], inplace=True, errors='ignore')
        
        # Criar gráfico mostrando as sessões ao longo do tempo
        fig = px.scatter(sessions, 
                        x=sessions['início'],  # Use a coluna já formatada
                        y='session_number', 
                        size='duração', 
                        color='is_logged_in',
                        color_discrete_map={True: '#4CAF50', False: '#9E9E9E'},
                        labels={'x': 'Data e Hora', 
                                'session_number': 'Número da Sessão',
                                'duração': 'Duração (s)',
                                'is_logged_in': 'Usuário Logado'},
                        hover_data={
                            'início': True,
                            'duração': True,
                            'session_source': True,
                            'session_medium': True
                        },
                        title="Histórico de Sessões do Usuário")
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir tabela clicável para selecionar sessão
        st.write("Clique em uma linha para ver os eventos dessa sessão:")
        st.dataframe(
            session_summary.style.apply(highlight_login, axis=1),
            use_container_width=True
        )
        
        # Selecionar sessão com dropdown
        session_numbers = user_data['session_number'].unique()
        # Adicionar opção para todas as sessões
        session_options = ["Todas as sessões"] + list(session_numbers)
        
        selection = st.selectbox(
            "Selecione uma sessão para ver os eventos:",
            session_options
        )
        
        if selection != "Todas as sessões":
            # Filtrar eventos da sessão selecionada
            session_events = user_data[user_data['session_number'] == selection].sort_values('event_timestamp')
            
            # Verificar se há um user_id nesta sessão
            user_id_in_session = session_events[session_events['user_id'].notna()]['user_id'].unique()
            user_id_display = user_id_in_session[0] if len(user_id_in_session) > 0 else "Não logado"
            
            st.subheader(f"Eventos da Sessão {selection}")
            st.markdown(f"""
            <div class='highlight-{'success' if session_events['is_logged_in'].any() else 'warning'}'>
                <p><b>Status:</b> {'Usuário logado' if session_events['is_logged_in'].any() else 'Usuário não logado'} |
                <b>User ID:</b> {user_id_display} |
                <b>Origem:</b> {session_events['session_source'].iloc[0]} |
                <b>Mídia:</b> {session_events['session_medium'].iloc[0]} |
                <b>Total de Eventos:</b> {len(session_events)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preparar os dados da sequência de eventos
            events_display = pd.DataFrame({
                'Evento': session_events['event_name'],
                'Timestamp': session_events['formatted_timestamp'],
                'Tempo desde início (s)': session_events['time_since_first_event'].round(1),
                'User ID': session_events['user_id'],
                'Usuário Logado': session_events['is_logged_in']
            })
            
            # Exibir tabela de eventos
            st.dataframe(
                events_display.style.apply(highlight_login, axis=1),
                use_container_width=True
            )
            
            # Visualização da sequência de eventos
            st.subheader("Sequência de Eventos na Sessão")
            
            # Criar linha do tempo de eventos
            fig = px.scatter(
                session_events,
                x='time_since_first_event',
                y='event_name',
                color='is_logged_in',
                size_max=10,
                color_discrete_map={True: '#4CAF50', False: '#9E9E9E'},
                labels={
                    'time_since_first_event': 'Tempo desde o início (s)',
                    'event_name': 'Evento',
                    'is_logged_in': 'Usuário Logado'
                }
            )

            # Adicionar linhas para conectar eventos em sequência
            events_ordered = session_events.sort_values('time_since_first_event')
            for i in range(len(events_ordered) - 1):
                fig.add_shape(
                    type="line",
                    x0=events_ordered.iloc[i]["time_since_first_event"],
                    y0=events_ordered.iloc[i]["event_name"],
                    x1=events_ordered.iloc[i+1]["time_since_first_event"],
                    y1=events_ordered.iloc[i+1]["event_name"],
                    line=dict(color="rgba(100, 100, 100, 0.5)", width=1)
                )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Opção para exportar eventos
            if st.button("Exportar eventos da sessão"):
                st.markdown(get_csv_download_link(events_display, 
                                                "eventos_sessao.csv", 
                                                "📥 Baixar eventos da sessão"), 
                          unsafe_allow_html=True)
        else:
            # Mostrar eventos de todas as sessões
            session_events = user_data.sort_values(['session_number', 'event_timestamp'])
            
            st.subheader("Eventos de Todas as Sessões")
            st.markdown(f"""
            <div class='highlight'>
                <p><b>Status:</b> {'Usuário logado em alguma sessão' if session_events['is_logged_in'].any() else 'Usuário não logado'} |
                <b>Total de Sessões:</b> {len(session_numbers)} |
                <b>Total de Eventos:</b> {len(session_events)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Usar abas para organizar as diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["Tabela de Eventos", "Eventos por Sessão", "Linha do Tempo"])
            
            with tab1:
                # Preparar os dados com a sessão incluída
                events_display = pd.DataFrame({
                    'Sessão': session_events['session_number'],
                    'Evento': session_events['event_name'],
                    'Timestamp': session_events['formatted_timestamp'],
                    'Tempo desde início (s)': session_events['time_since_first_event'].round(1),
                    'User ID': session_events['user_id'],
                    'Usuário Logado': session_events['is_logged_in']
                })
                
                st.dataframe(
                    events_display.style.apply(highlight_login, axis=1),
                    use_container_width=True
                )
                
                # Opção para exportar todos os eventos
                if st.button("Exportar todos os eventos"):
                    st.markdown(get_csv_download_link(events_display, 
                                                    "todos_eventos_usuario.csv", 
                                                    "📥 Baixar todos os eventos"), 
                              unsafe_allow_html=True)
            
            with tab2:
                # Visualização da sequência de eventos por sessão
                session_events_plot = session_events.copy()
                session_events_plot['session_number'] = session_events_plot['session_number'].astype(str)
                
                fig = px.scatter(
                    session_events_plot,
                    x='time_since_first_event',
                    y='event_name',
                    color='session_number',
                    size_max=10,
                    labels={
                        'time_since_first_event': 'Tempo desde o início da sessão (s)',
                        'event_name': 'Evento',
                        'session_number': 'Sessão'
                    }
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Linha do tempo cronológica
                session_events_plot['datetime'] = pd.to_datetime(session_events_plot['event_timestamp']/1000000, unit='s')
                
                fig2 = px.scatter(
                    session_events_plot,
                    x='datetime',
                    y='session_number',
                    color='event_name',
                    size_max=10,
                    labels={
                        'datetime': 'Data e Hora',
                        'session_number': 'Sessão',
                        'event_name': 'Evento'
                    }
                )
                
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

# Página de análise de eventos
def show_event_analysis_page(filtered_df):
    st.header("📊 Análise de Eventos")
    
    # Obter todos os eventos disponíveis e suas contagens
    all_events = sorted(filtered_df['event_name'].unique())
    event_counts = filtered_df['event_name'].value_counts()
    
    # Interface principal com abas
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visão Geral", "⏰ Distribuição Temporal", "🔍 Análise Detalhada", "🔄 Eventos por Sessão"])
    
    with tab1:
        st.subheader("Visão Geral dos Eventos")
        
        # Top eventos
        top_n = st.slider("Número de eventos para exibir:", 5, 30, 10)
        
        # Exibir gráfico de barras com top eventos
        top_events = event_counts.head(top_n)
        fig = px.bar(
            top_events, 
            labels={'index': 'Evento', 'value': 'Contagem'},
            title=f"Top {top_n} Eventos Mais Frequentes"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas resumidas
        total_events = len(filtered_df)
        unique_events = len(all_events)
        total_sessions = filtered_df['session_number'].nunique()
        total_users = filtered_df['client_id'].nunique()
        
        # Exibir métricas em colunas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Eventos", f"{total_events:,}")
        with col2:
            st.metric("Tipos de Eventos", f"{unique_events}")
        with col3:
            st.metric("Total de Sessões", f"{total_sessions:,}")
        with col4:
            st.metric("Usuários Únicos", f"{total_users:,}")
            
        # Tabela com detalhes dos eventos
        st.subheader("Detalhes dos Eventos")
        events_df = pd.DataFrame({
            'Evento': event_counts.index,
            'Contagem': event_counts.values,
            'Porcentagem': (event_counts.values / total_events * 100).round(2)
        })
        
        st.dataframe(events_df, use_container_width=True)
        
        # Download dos dados
        st.markdown(get_csv_download_link(events_df, 
                                         "eventos_resumo.csv",
                                         "📥 Baixar resumo de eventos"), 
                    unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Distribuição Temporal dos Eventos")
        
        # Seleção de eventos para análise temporal
        selected_events = st.multiselect(
            "Selecione eventos para análise temporal:",
            options=all_events,
            default=all_events[:min(5, len(all_events))],
            format_func=lambda x: f"{x} ({event_counts[x]})"
        )
        
        if selected_events:
            # Seleção do tipo de agregação temporal
            time_agg = st.radio(
                "Agregar por:",
                ["Dia", "Hora do Dia", "Dia da Semana"]
            )
            
            # Preparar dados conforme agregação
            if time_agg == "Dia":
                # Agrupar por data e tipo de evento
                time_data = filtered_df[filtered_df['event_name'].isin(selected_events)].copy()
                events_by_date = time_data.groupby(['date', 'event_name']).size().reset_index(name='count')
                
                # Criar gráfico de linha por data
                fig = px.line(
                    events_by_date, 
                    x='date', 
                    y='count', 
                    color='event_name',
                    labels={'date': 'Data', 'count': 'Contagem', 'event_name': 'Evento'},
                    title="Eventos por Dia"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            elif time_agg == "Hora do Dia":
                # Agrupar por hora do dia e tipo de evento
                time_data = filtered_df[filtered_df['event_name'].isin(selected_events)].copy()
                events_by_hour = time_data.groupby(['hour', 'event_name']).size().reset_index(name='count')
                
                # Criar gráfico de linha por hora
                fig = px.line(
                    events_by_hour, 
                    x='hour', 
                    y='count', 
                    color='event_name',
                    labels={'hour': 'Hora do Dia', 'count': 'Contagem', 'event_name': 'Evento'},
                    title="Eventos por Hora do Dia"
                )
                
                fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de hora do dia x evento
                pivot_hour = events_by_hour.pivot(index='event_name', columns='hour', values='count').fillna(0)
                
                fig = px.imshow(
                    pivot_hour,
                    labels=dict(x="Hora do Dia", y="Evento", color="Contagem"),
                    title="Heatmap de Eventos por Hora"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Dia da Semana
                # Agrupar por dia da semana e tipo de evento
                time_data = filtered_df[filtered_df['event_name'].isin(selected_events)].copy()
                events_by_weekday = time_data.groupby(['weekday', 'event_name']).size().reset_index(name='count')
                
                # Ordenar dias da semana
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                events_by_weekday['weekday_idx'] = events_by_weekday['weekday'].apply(lambda x: weekday_order.index(x))
                events_by_weekday = events_by_weekday.sort_values('weekday_idx')
                
                # Criar gráfico de barras por dia da semana
                fig = px.bar(
                    events_by_weekday, 
                    x='weekday', 
                    y='count', 
                    color='event_name',
                    labels={'weekday': 'Dia da Semana', 'count': 'Contagem', 'event_name': 'Evento'},
                    title="Eventos por Dia da Semana",
                    category_orders={"weekday": weekday_order}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            # Download dos dados
            st.markdown(get_csv_download_link(events_by_date if time_agg == "Dia" else 
                                             events_by_hour if time_agg == "Hora do Dia" else
                                             events_by_weekday, 
                                         f"eventos_{time_agg.lower()}.csv",
                                         f"📥 Baixar dados por {time_agg}"), 
                      unsafe_allow_html=True)
        else:
            st.warning("Selecione pelo menos um evento para análise temporal.")
    
    with tab3:
        st.subheader("Análise Detalhada de Eventos")
        
        # Seleção de evento para análise detalhada
        event_for_analysis = st.selectbox(
            "Selecione um evento para análise detalhada:",
            options=all_events,
            format_func=lambda x: f"{x} ({event_counts[x]})"
        )
        
        if event_for_analysis:
            # Filtrar dados para o evento selecionado
            event_data = filtered_df[filtered_df['event_name'] == event_for_analysis].copy()
            
            # Métricas específicas do evento
            event_sessions = event_data['session_number'].nunique()
            event_users = event_data['client_id'].nunique()
            event_logged_in = event_data[event_data['is_logged_in']]['client_id'].nunique()
            event_not_logged_in = event_users - event_logged_in
            
            # Exibir métricas em colunas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Ocorrências", f"{len(event_data):,}")
            with col2:
                st.metric("Sessões Únicas", f"{event_sessions:,}")
            with col3:
                st.metric("Usuários Únicos", f"{event_users:,}")
            with col4:
                st.metric("% Usuários Logados", f"{event_logged_in / event_users * 100:.1f}%" if event_users > 0 else "0%")
            
            # Análise por origem de tráfego
            st.subheader(f"Distribuição por Origem de Tráfego - {event_for_analysis}")
            
            # Agrupar por origem/mídia
            traffic_data = event_data.groupby(['session_source', 'session_medium']).size().reset_index(name='count')
            traffic_data['source_medium'] = traffic_data['session_source'] + ' / ' + traffic_data['session_medium']
            traffic_data = traffic_data.sort_values('count', ascending=False)
            
            # Mostrar gráfico top origens
            top_traffic = traffic_data.head(10)
            
            if not top_traffic.empty:
                fig = px.pie(
                    top_traffic,
                    values='count',
                    names='source_medium',
                    title="Top 10 Origens de Tráfego"
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlação com outros eventos
            st.subheader(f"Eventos Relacionados a {event_for_analysis}")
            
            # Encontrar outros eventos que ocorrem nas mesmas sessões
            event_sessions = set(event_data[['client_id', 'session_number']].itertuples(index=False, name=None))
            
            related_events = []
            for other_event in all_events:
                if other_event != event_for_analysis:
                    other_data = filtered_df[filtered_df['event_name'] == other_event]
                    other_sessions = set(other_data[['client_id', 'session_number']].itertuples(index=False, name=None))
                    
                    # Sessões que têm ambos os eventos
                    common_sessions = event_sessions.intersection(other_sessions)
                    
                    if common_sessions:
                        related_events.append({
                            'Evento': other_event,
                            'Sessões em Comum': len(common_sessions),
                            'Porcentagem': len(common_sessions) / len(event_sessions) * 100
                        })
            
            if related_events:
                # Criar DataFrame
                related_df = pd.DataFrame(related_events)
                related_df = related_df.sort_values('Sessões em Comum', ascending=False)
                
                # Mostrar top eventos relacionados
                top_related = related_df.head(15)
                
                fig = px.bar(
                    top_related,
                    x='Porcentagem',
                    y='Evento',
                    orientation='h',
                    labels={
                        'Porcentagem': '% de Sessões em Comum',
                        'Evento': 'Evento Relacionado'
                    },
                    title=f"Eventos que Ocorrem nas Mesmas Sessões que {event_for_analysis}"
                )
                
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportar dados
                st.markdown(get_csv_download_link(related_df, 
                                                f"eventos_relacionados_{event_for_analysis}.csv",
                                                "📥 Baixar análise de eventos relacionados"), 
                          unsafe_allow_html=True)
            else:
                st.info("Não foram encontrados eventos relacionados.")
    
    with tab4:
        st.subheader("🔄 Análise de Eventos por Sessão")
        
        # Seleção de evento para análise
        evento_analise = st.selectbox(
            "Selecione um evento para analisar:",
            options=all_events,
            format_func=lambda x: f"{x} ({event_counts[x]})"
        )
        
        if evento_analise:
            # Identificar em quais sessões este evento ocorre
            sessoes_com_evento = filtered_df[filtered_df['event_name'] == evento_analise]['session_number'].unique()
            max_sessao = int(sessoes_com_evento.max()) if len(sessoes_com_evento) > 0 else 10
            
            # Controles para seleção de sessão
            col1, col2 = st.columns(2)
            
            with col1:
                # Seleção de número da sessão
                sessao_selecionada = st.slider(
                    "Número da sessão para análise:",
                    min_value=1,
                    max_value=max_sessao,
                    value=min(4, max_sessao)
                )
            
            with col2:
                # Opção para filtrar apenas primeiras ocorrências
                apenas_primeiro_disparo = st.checkbox(
                    "Mostrar apenas usuários que dispararam este evento pela primeira vez nesta sessão",
                    value=False,
                    help="Se marcado, vai filtrar apenas os usuários que nunca dispararam este evento em sessões anteriores"
                )
            
            # Obter todos os usuários que dispararam o evento na sessão selecionada
            usuarios_com_evento_na_sessao = filtered_df[
                (filtered_df['event_name'] == evento_analise) & 
                (filtered_df['session_number'] == sessao_selecionada)
            ]['client_id'].unique()
            
            if len(usuarios_com_evento_na_sessao) > 0:
                # Se opção de primeiro disparo estiver marcada, filtrar mais
                if apenas_primeiro_disparo:
                    # Para cada usuário, verificar se ele disparou o evento em sessões anteriores
                    usuarios_primeiro_disparo = []
                    
                    for user_id in usuarios_com_evento_na_sessao:
                        # Obter todas as ocorrências do evento para este usuário
                        ocorrencias_usuario = filtered_df[
                            (filtered_df['client_id'] == user_id) & 
                            (filtered_df['event_name'] == evento_analise)
                        ]
                        
                        # Verificar se a primeira ocorrência é na sessão selecionada
                        primeira_sessao = ocorrencias_usuario['session_number'].min()
                        if primeira_sessao == sessao_selecionada:
                            usuarios_primeiro_disparo.append(user_id)
                    
                    # Atualizar a lista de usuários para análise
                    usuarios_para_analise = usuarios_primeiro_disparo
                    st.success(f"Encontrados {len(usuarios_para_analise)} usuários que dispararam '{evento_analise}' pela primeira vez na sessão {sessao_selecionada}")
                else:
                    usuarios_para_analise = usuarios_com_evento_na_sessao
                    st.success(f"Encontrados {len(usuarios_para_analise)} usuários que dispararam '{evento_analise}' na sessão {sessao_selecionada}")
                
                # Se temos usuários para analisar, seguir com a análise
                if len(usuarios_para_analise) > 0:
                    # Obter todas as sessões desses usuários até a sessão selecionada
                    dados_jornada = filtered_df[
                        (filtered_df['client_id'].isin(usuarios_para_analise)) & 
                        (filtered_df['session_number'] <= sessao_selecionada)
                    ].copy()
                    
                    # Agrupar por usuário e sessão para obter a origem/mídia de cada sessão
                    origens_por_sessao = dados_jornada.groupby(['client_id', 'session_number']).agg({
                        'session_source': 'first',
                        'session_medium': 'first'
                    }).reset_index()
                    
                    # Concatenar origem e mídia
                    origens_por_sessao['origem_midia'] = origens_por_sessao['session_source'].fillna('(none)') + '/' + origens_por_sessao['session_medium'].fillna('(none)')
                    
                    # Analisar a distribuição de origens/mídias por número da sessão
                    distribuicao_origens = origens_por_sessao.groupby(['session_number', 'origem_midia']).size().reset_index(name='count')
                    
                    # Calcular percentuais por sessão
                    total_por_sessao = distribuicao_origens.groupby('session_number')['count'].sum().reset_index()
                    distribuicao_origens = pd.merge(distribuicao_origens, total_por_sessao, on='session_number', suffixes=('', '_total'))
                    distribuicao_origens['percentual'] = (distribuicao_origens['count'] / distribuicao_origens['count_total'] * 100).round(1)
                    
                    # Ordenar para visualização
                    distribuicao_origens = distribuicao_origens.sort_values(['session_number', 'count'], ascending=[True, False])
                    
                    # Visualizar a jornada através das sessões
                    st.subheader(f"Jornada de Origens/Mídias até o Evento '{evento_analise}' na Sessão {sessao_selecionada}")
                    
                    # Interface para escolher visualização
                    formato_viz = st.radio(
                        "Formato de visualização:",
                        ["Gráfico de Barras", "Gráfico de Área", "Tabela Detalhada"],
                        horizontal=True
                    )
                    
                    if formato_viz == "Gráfico de Barras":
                        # Mostrar top origens por sessão
                        fig = px.bar(
                            distribuicao_origens,
                            x='session_number',
                            y='percentual',
                            color='origem_midia',
                            labels={
                                'session_number': 'Número da Sessão',
                                'percentual': 'Porcentagem de Usuários (%)',
                                'origem_midia': 'Origem/Mídia'
                            },
                            title=f"Distribuição de Origens/Mídias por Sessão para Usuários com '{evento_analise}' na Sessão {sessao_selecionada}"
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickmode='linear', dtick=1),
                            height=500,
                            legend_title_text='Origem/Mídia'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif formato_viz == "Gráfico de Área":
                        # Preparar dados para gráfico de área empilhada
                        # Precisamos de um pivot
                        pivot_origens = distribuicao_origens.pivot(index='session_number', columns='origem_midia', values='percentual').fillna(0)
                        
                        # Converter para formato long para plotly
                        long_data = pivot_origens.reset_index().melt(
                            id_vars='session_number',
                            value_vars=pivot_origens.columns,
                            var_name='origem_midia',
                            value_name='percentual'
                        )
                        
                        # Ordenar por sessão
                        long_data = long_data.sort_values('session_number')
                        
                        # Criar gráfico de área
                        fig = px.area(
                            long_data,
                            x='session_number',
                            y='percentual',
                            color='origem_midia',
                            labels={
                                'session_number': 'Número da Sessão',
                                'percentual': 'Porcentagem de Usuários (%)',
                                'origem_midia': 'Origem/Mídia'
                            },
                            title=f"Tendência de Origens/Mídias por Sessão para Usuários com '{evento_analise}' na Sessão {sessao_selecionada}"
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickmode='linear', dtick=1),
                            height=500,
                            legend_title_text='Origem/Mídia'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:  # Tabela detalhada
                        # Mostrar tabela com porcentagens
                        tabela_origens = distribuicao_origens[['session_number', 'origem_midia', 'count', 'percentual']]
                        tabela_origens.columns = ['Sessão', 'Origem/Mídia', 'Usuários', 'Porcentagem (%)']
                        
                        st.dataframe(tabela_origens, use_container_width=True)
                    
                    # Análise das sequências de origens mais comuns
                    st.subheader("Sequências de Origens/Mídias Mais Comuns")
                    
                    # Para cada usuário, mapear sua sequência de origens
                    usuarios_sequencias = {}
                    for user_id in usuarios_para_analise:
                        # Obter sessões do usuário em ordem
                        sessoes_usuario = origens_por_sessao[
                            origens_por_sessao['client_id'] == user_id
                        ].sort_values('session_number')
                        
                        # Criar sequência
                        sequencia = []
                        for i in range(1, sessao_selecionada + 1):
                            sessao_atual = sessoes_usuario[sessoes_usuario['session_number'] == i]
                            if not sessao_atual.empty:
                                sequencia.append(sessao_atual['origem_midia'].iloc[0])
                            else:
                                sequencia.append("(sem dados)")
                        
                        # Guardar sequência
                        usuarios_sequencias[user_id] = sequencia
                    
                    # Contar frequência de cada sequência
                    sequencias_count = {}
                    for seq in usuarios_sequencias.values():
                        seq_str = " → ".join(seq)
                        if seq_str in sequencias_count:
                            sequencias_count[seq_str] += 1
                        else:
                            sequencias_count[seq_str] = 1
                    
                    # Converter para DataFrame
                    sequencias_df = pd.DataFrame({
                        'Sequência': list(sequencias_count.keys()),
                        'Contagem': list(sequencias_count.values())
                    })
                    
                    # Calcular percentual
                    total_usuarios = len(usuarios_para_analise)
                    sequencias_df['Porcentagem (%)'] = (sequencias_df['Contagem'] / total_usuarios * 100).round(1)
                    
                    # Ordenar
                    sequencias_df = sequencias_df.sort_values('Contagem', ascending=False)
                    
                    # Mostrar top 10 sequências
                    top_n_sequencias = min(10, len(sequencias_df))
                    
                    if top_n_sequencias > 0:
                        # Visualização das top sequências
                        fig = px.bar(
                            sequencias_df.head(top_n_sequencias),
                            x='Porcentagem (%)',
                            y='Sequência',
                            orientation='h',
                            labels={
                                'Porcentagem (%)': 'Porcentagem de Usuários (%)',
                                'Sequência': 'Sequência de Origem/Mídia'
                            },
                            title=f"Top {top_n_sequencias} Sequências de Origens/Mídias para Usuários com '{evento_analise}' na Sessão {sessao_selecionada}"
                        )
                        
                        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabela completa
                        st.subheader("Todas as Sequências de Origens/Mídias")
                        st.dataframe(sequencias_df, use_container_width=True)
                        
                        # Exportar dados
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(get_csv_download_link(
                                distribuicao_origens, 
                                f"origens_para_{evento_analise}_sessao_{sessao_selecionada}.csv",
                                "📥 Baixar dados de distribuição de origens"
                            ), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(get_csv_download_link(
                                sequencias_df, 
                                f"sequencias_para_{evento_analise}_sessao_{sessao_selecionada}.csv",
                                "📥 Baixar dados de sequências"
                            ), unsafe_allow_html=True)
                    else:
                        st.warning("Não foi possível identificar sequências comuns com os dados disponíveis.")
                else:
                    st.warning(f"Não foram encontrados usuários que dispararam '{evento_analise}' pela primeira vez na sessão {sessao_selecionada}.")
            else:
                st.warning(f"Não foram encontrados usuários que dispararam '{evento_analise}' na sessão {sessao_selecionada}.")

# Página de análise de jornada do usuário
def show_user_journey_page(filtered_df):
    st.header("📈 Análise de Jornada do Usuário")
    
    # Verificação de dependências para recursos avançados
    missing_deps = []
    if not optional_deps_available.get('networkx', False):
        missing_deps.append('networkx')
    if not optional_deps_available.get('wordcloud', False):
        missing_deps.append('wordcloud')
    
    if missing_deps:
        st.warning(f"⚠️ Algumas visualizações avançadas estão desabilitadas. Para habilitar, instale: {', '.join(missing_deps)}")
    
    # Obter todos os eventos disponíveis e suas contagens
    all_events = sorted(filtered_df['event_name'].unique())
    event_counts = filtered_df['event_name'].value_counts()
    
    # Filtro global de eventos para toda a página de jornada do usuário
    with st.expander("🔍 Filtrar eventos para análise", expanded=True):
        # Identificar eventos potencialmente automáticos
        auto_events = [e for e in all_events if any(keyword in e.lower() for keyword in 
                                                 ['page_view', 'session', 'scroll', 'user_engagement', 
                                                  'click', 'view_', 'scroll', 'timing', 'gtm', 'firebase', 
                                                  'first_visit', 'app_'])]
        
        # Outros eventos (potencialmente mais importantes)
        other_events = [e for e in all_events if e not in auto_events]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Opção para selecionar todos ou nenhum evento
            select_all = st.checkbox("Selecionar todos os eventos", value=True)
            # Opção para excluir eventos automáticos
            exclude_auto = st.checkbox("Excluir eventos automáticos", value=False)
        
        with col2:
            filter_by = st.radio(
                "Método de filtragem:",
                ["Selecionar eventos para incluir", "Selecionar eventos para excluir"],
                index=0
            )
        
        # Lista de eventos com contagem
        event_list = [(f"{e} ({event_counts[e]})", e) for e in all_events]
        event_list.sort(key=lambda x: x[0])
        
        # Determinar seleção padrão baseada nos checkboxes
        if select_all:
            default_selection = all_events
        elif exclude_auto:
            default_selection = other_events
        else:
            default_selection = []
        
        # Widget para selecionar eventos
        if filter_by == "Selecionar eventos para incluir":
            selected_events = st.multiselect(
                "Selecione os eventos para incluir na análise:",
                options=all_events,
                default=default_selection,
                format_func=lambda x: f"{x} ({event_counts[x]})"
            )
            events_to_use = selected_events
        else:
            excluded_events = st.multiselect(
                "Selecione os eventos para excluir da análise:",
                options=all_events,
                default=[] if select_all else auto_events,
                format_func=lambda x: f"{x} ({event_counts[x]})"
            )
            events_to_use = [e for e in all_events if e not in excluded_events]
        
        # Aplicar filtro na DataFrame
        if events_to_use:
            journey_df = filtered_df[filtered_df['event_name'].isin(events_to_use)].copy()
            st.success(f"✅ Analisando {len(events_to_use)} eventos selecionados ({len(journey_df)} registros)")
        else:
            st.error("⚠️ Nenhum evento selecionado. Por favor, selecione pelo menos um evento para análise.")
            journey_df = filtered_df.copy()  # Usar todos os eventos como fallback
    
    # Layout principal das abas - usar journey_df (filtrada) em vez de filtered_df
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛤️ Caminhos de Navegação", 
        "⏱️ Tempos de Conversão", 
        "🔍 Análise de Funil", 
        "🔄 Fluxo entre Eventos"
    ])
    
    with tab1:
        st.subheader("Caminhos de Navegação Mais Frequentes")
        
        # Seleção de número de passos e eventos
        col1, col2 = st.columns(2)
        with col1:
            path_length = st.slider("Número de passos no caminho:", 2, 6, 3)
        with col2:
            top_paths = st.slider("Mostrar top caminhos:", 5, 30, 10)
        
        # O multiselect de exclusão pode ser mantido para filtros adicionais específicos desta aba
        exclude_events = st.multiselect(
            "Excluir eventos adicionais desta análise específica (opcional):",
            options=sorted(journey_df['event_name'].unique()),
            default=[]
        )
        
        if exclude_events:
            path_df = journey_df[~journey_df['event_name'].isin(exclude_events)].copy()
        else:
            path_df = journey_df.copy()
        
        # Implementação para encontrar caminhos frequentes
        if not path_df.empty:
            # Agrupar por client_id e session_number e criar uma lista de eventos em ordem
            paths_by_session = path_df.sort_values(['client_id', 'session_number', 'event_timestamp'])\
                                .groupby(['client_id', 'session_number'])['event_name'].agg(list).reset_index()
            
            # Função para extrair subsequências de um caminho
            def extract_subsequences(event_list, length):
                return [tuple(event_list[i:i+length]) for i in range(len(event_list) - length + 1)]
            
            # Extrair todos os caminhos da extensão solicitada
            all_paths = []
            for path in paths_by_session['event_name']:
                if len(path) >= path_length:
                    all_paths.extend(extract_subsequences(path, path_length))
            
            if all_paths:
                # Contar frequência dos caminhos
                path_counts = pd.Series(all_paths).value_counts().reset_index()
                path_counts.columns = ['Caminho', 'Frequência']
                
                # Converter tuplas para strings mais legíveis
                path_counts['Caminho'] = path_counts['Caminho'].apply(lambda x: ' → '.join(x))
                
                # Mostrar os caminhos mais frequentes
                st.subheader(f"Top {top_paths} caminhos mais frequentes (sequência de {path_length} eventos)")
                
                # Filtrar para top caminhos
                top_path_counts = path_counts.head(top_paths)
                
                # Criar gráfico
                fig = px.bar(
                    top_path_counts, 
                    x='Frequência', 
                    y='Caminho',
                    orientation='h',
                    labels={'Frequência': 'Número de Ocorrências', 'Caminho': 'Sequência de Eventos'},
                    title=f"Caminhos de navegação mais frequentes (sequência de {path_length} eventos)"
                )
                
                fig.update_layout(height=max(400, 30*len(top_path_counts)), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportar dados
                st.markdown("##### Exportar dados de caminhos")
                st.markdown(get_csv_download_link(path_counts, 
                                                f"caminhos_navegacao_{path_length}_passos.csv",
                                                "📥 Baixar dados de caminhos"), 
                         unsafe_allow_html=True)
            else:
                st.warning(f"Não foram encontrados caminhos com {path_length} eventos consecutivos. Tente reduzir o tamanho do caminho.")
        else:
            st.error("Não há dados suficientes para análise após aplicar os filtros.")
    
    with tab2:
        st.subheader("Tempo de Conversão entre Eventos")
        
        # Interface para selecionar eventos de início e fim
        available_events = sorted(journey_df['event_name'].unique())
        
        if len(available_events) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                start_event = st.selectbox(
                    "Evento inicial:",
                    options=available_events,
                    index=0 if available_events else None
                )
            
            with col2:
                # Filtra para não permitir selecionar o mesmo evento inicial
                end_events = [e for e in available_events if e != start_event]
                end_event = st.selectbox(
                    "Evento final:",
                    options=end_events,
                    index=0 if end_events else None
                )
            
            # Opções adicionais
            col1, col2 = st.columns(2)
            with col1:
                segment_by = st.radio(
                    "Segmentar por:",
                    ["Nenhum", "Status de Login", "Origem/Mídia"]
                )
            
            with col2:
                max_time = st.slider(
                    "Tempo máximo para conversão (minutos):",
                    1, 60, 30
                )
            
            if st.button("Analisar Tempos de Conversão"):
                # Filtrar apenas sessões que têm ambos os eventos
                sessions_with_both = journey_df[journey_df['event_name'].isin([start_event, end_event])]\
                                     .groupby(['client_id', 'session_number'])['event_name']\
                                     .unique()\
                                     .apply(lambda x: set([start_event, end_event]).issubset(set(x)))
                
                valid_sessions = sessions_with_both[sessions_with_both].index.tolist()
                
                if valid_sessions:
                    # Dados para análise
                    conversion_data = journey_df[
                        (journey_df['event_name'].isin([start_event, end_event])) & 
                        (journey_df.set_index(['client_id', 'session_number']).index.isin(valid_sessions))
                    ].copy()
                    
                    # Calcular tempo entre eventos para cada sessão
                    conversion_times = []
                    
                    for (client, session), group in conversion_data.groupby(['client_id', 'session_number']):
                        # Obter o primeiro evento de início e o primeiro de fim
                        start_time = group[group['event_name'] == start_event]['event_timestamp'].min()
                        end_time = group[group['event_name'] == end_event]['event_timestamp'].min()
                        
                        # Verificar se o evento de fim ocorre após o de início
                        if end_time > start_time:
                            time_diff = (end_time - start_time) / 1000000  # Converter para segundos
                            
                            # Filtrar por tempo máximo (converter minutos para segundos)
                            if time_diff <= (max_time * 60):
                                # Adicionar dados para segmentação
                                is_logged_in = group['is_logged_in'].any()
                                source = group['session_source'].iloc[0]
                                medium = group['session_medium'].iloc[0]
                                
                                conversion_times.append({
                                    'client_id': client,
                                    'session_number': session,
                                    'time_diff_seconds': time_diff,
                                    'is_logged_in': is_logged_in,
                                    'source': source,
                                    'medium': medium,
                                    'source_medium': f"{source}/{medium}" if source and medium else "Não definido"
                                })
                    
                    if conversion_times:
                        # Criar DataFrame com tempos
                        times_df = pd.DataFrame(conversion_times)
                        
                        # Mostrar estatísticas gerais
                        avg_time = times_df['time_diff_seconds'].mean()
                        median_time = times_df['time_diff_seconds'].median()
                        min_time = times_df['time_diff_seconds'].min()
                        max_time = times_df['time_diff_seconds'].max()
                        
                        # Formatação para exibição
                        fmt_time = lambda secs: f"{int(secs//60)}m {int(secs%60)}s"
                        
                        # Exibir estatísticas
                        st.markdown(f"""
                        #### Estatísticas de tempo entre **{start_event}** e **{end_event}**
                        
                        - **Total de conversões:** {len(times_df)}
                        - **Tempo médio:** {fmt_time(avg_time)} ({avg_time:.1f}s)
                        - **Tempo mediano:** {fmt_time(median_time)} ({median_time:.1f}s)
                        - **Tempo mínimo:** {fmt_time(min_time)} ({min_time:.1f}s)
                        - **Tempo máximo:** {fmt_time(max_time)} ({max_time:.1f}s)
                        """)
                        
                        # Criar histograma de distribuição
                        fig = px.histogram(
                            times_df, 
                            x='time_diff_seconds',
                            nbins=20,
                            labels={'time_diff_seconds': 'Tempo (segundos)'},
                            title=f"Distribuição do tempo entre {start_event} e {end_event}"
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Segmentação por atributo selecionado
                        if segment_by == "Status de Login":
                            # Agrupar por status de login
                            login_stats = times_df.groupby('is_logged_in')['time_diff_seconds'].agg(
                                ['mean', 'median', 'min', 'max', 'count']
                            ).reset_index()
                            
                            login_stats.columns = ['Usuário Logado', 'Média (s)', 'Mediana (s)', 'Mínimo (s)', 'Máximo (s)', 'Contagem']
                            
                            # Formatar booleano
                            login_stats['Usuário Logado'] = login_stats['Usuário Logado'].map({True: 'Sim', False: 'Não'})
                            
                            st.subheader("Segmentação por Status de Login")
                            st.dataframe(login_stats)
                            
                            # Gráfico de comparação
                            fig = px.box(
                                times_df,
                                x='is_logged_in',
                                y='time_diff_seconds',
                                labels={
                                    'is_logged_in': 'Usuário Logado',
                                    'time_diff_seconds': 'Tempo (segundos)'
                                },
                                category_orders={'is_logged_in': [True, False]},
                                title="Comparação de tempos por status de login"
                            )
                            
                            # Atualizar nomes das categorias
                            fig.update_xaxes(
                                ticktext=["Logado", "Não Logado"],
                                tickvals=[True, False]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif segment_by == "Origem/Mídia":
                            # Top origens/mídias
                            top_sources = times_df['source_medium'].value_counts().head(10).index.tolist()
                            
                            # Filtrar para top origens
                            source_data = times_df[times_df['source_medium'].isin(top_sources)].copy()
                            
                            if not source_data.empty:
                                # Agrupar por origem/mídia
                                source_stats = source_data.groupby('source_medium')['time_diff_seconds'].agg(
                                    ['mean', 'median', 'min', 'max', 'count']
                                ).reset_index()
                                
                                source_stats.columns = ['Origem/Mídia', 'Média (s)', 'Mediana (s)', 'Mínimo (s)', 'Máximo (s)', 'Contagem']
                                
                                # Ordenar por contagem
                                source_stats = source_stats.sort_values('Contagem', ascending=False)
                                
                                st.subheader("Segmentação por Origem/Mídia (Top 10)")
                                st.dataframe(source_stats)
                                
                                # Gráfico de comparação
                                if len(source_data) >= 5:  # Só mostrar se tiver dados suficientes
                                    fig = px.box(
                                        source_data,
                                        x='source_medium',
                                        y='time_diff_seconds',
                                        labels={
                                            'source_medium': 'Origem/Mídia',
                                            'time_diff_seconds': 'Tempo (segundos)'
                                        },
                                        title="Comparação de tempos por origem/mídia"
                                    )
                                    
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Dados insuficientes para segmentação por origem/mídia.")
                        
                        # Exportar dados
                        st.markdown("##### Exportar dados de conversão")
                        st.markdown(get_csv_download_link(times_df, 
                                                    f"tempos_conversao_{start_event}_to_{end_event}.csv",
                                                    "📥 Baixar dados de tempos"), 
                                 unsafe_allow_html=True)
                    else:
                        st.warning(f"Não foram encontradas conversões de {start_event} para {end_event} dentro do limite de tempo de {max_time} minutos.")
                else:
                    st.warning(f"Não foram encontradas sessões que contenham ambos eventos: {start_event} e {end_event}.")
        else:
            st.error("Não há eventos suficientes para análise. Selecione pelo menos dois eventos diferentes.")
    
    with tab3:
        st.subheader("Análise de Funil")
        
        # Interface para criar um funil personalizado
        st.markdown("Defina uma sequência de eventos para analisar como funil de conversão.")
        
        # Lista de eventos disponíveis (já filtrados)
        available_events = sorted(journey_df['event_name'].unique())
        
        if len(available_events) >= 2:
            # Seleção de eventos múltiplos para o funil
            funnel_events = st.multiselect(
                "Selecione os eventos do funil (na ordem desejada):",
                options=available_events,
                default=available_events[:min(3, len(available_events))]
            )
            
            # Opções de análise
            col1, col2 = st.columns(2)
            
            with col1:
                strict_order = st.checkbox("Ordem estrita dos eventos", value=True, 
                                          help="Se marcado, os eventos devem ocorrer na ordem exata. Se desmarcado, apenas verifica se ocorreram na sessão.")
                
            with col2:
                session_based = st.checkbox("Análise por sessão", value=True,
                                          help="Se marcado, analisa conversões dentro de cada sessão. Se desmarcado, analisa conversões em todas as sessões do usuário.")
            
            # Verificar se há eventos suficientes
            if len(funnel_events) >= 2:
                if st.button("Gerar Funil"):
                    # Preparar os dados conforme as opções
                    if session_based:
                        # Analisar por sessão
                        groupby_cols = ['client_id', 'session_number']
                    else:
                        # Analisar por usuário em todas as sessões
                        groupby_cols = ['client_id']
                    
                    if strict_order:
                        # Funil com ordem estrita - implementação já existe na função create_funnel_chart
                        funnel_df = journey_df[journey_df['event_name'].isin(funnel_events)].copy()
                        fig, funnel_data = create_funnel_chart(funnel_df, funnel_events, 
                                                              title=f"Funil de Conversão - {'Sessão' if session_based else 'Usuário'}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabela com dados detalhados
                        st.subheader("Detalhes do Funil")
                        st.dataframe(funnel_data)
                        
                    else:
                        # Versão simplificada: apenas verifica se os eventos ocorreram, sem ordem
                        # Contar quantos usuários/sessões realizaram cada evento
                        funnel_data = []
                        
                        for event in funnel_events:
                            # Filtrar para este evento
                            event_data = journey_df[journey_df['event_name'] == event]
                            
                            # Contar usuários/sessões únicos
                            unique_count = event_data.groupby(groupby_cols).size().shape[0]
                            
                            funnel_data.append({
                                'Evento': event,
                                'Usuários/Sessões': unique_count
                            })
                        
                        # Criar DataFrame
                        funnel_df = pd.DataFrame(funnel_data)
                        
                        # Calcular taxa de conversão
                        max_count = funnel_df['Usuários/Sessões'].max()
                        funnel_df['Taxa (%)'] = funnel_df['Usuários/Sessões'] / max_count * 100
                        
                        # Criar gráfico de funil
                        fig = px.funnel(
                            funnel_df,
                            x='Usuários/Sessões',
                            y='Evento',
                            title=f"Funil de Eventos - {'Sessão' if session_based else 'Usuário'} (Sem ordem estrita)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabela com dados detalhados
                        st.subheader("Detalhes do Funil")
                        st.dataframe(funnel_df)
                    
                    # Exportar dados
                    st.markdown("##### Exportar dados do funil")
                    st.markdown(get_csv_download_link(funnel_data if strict_order else funnel_df, 
                                                  "dados_funil.csv",
                                                  "📥 Baixar dados do funil"), 
                               unsafe_allow_html=True)
                else:
                    st.warning("Selecione pelo menos 2 eventos para gerar um funil de conversão.")
            else:
                st.warning("Não há eventos suficientes para análise. Selecione pelo menos dois eventos diferentes.")
        else:
            st.error("Não há eventos suficientes para análise. Selecione pelo menos dois eventos diferentes.")
    
    with tab4:
        st.subheader("Fluxo entre Eventos")
        
        # Verificar se a biblioteca networkx está disponível
        if not optional_deps_available.get('networkx', False):
            st.error("Esta funcionalidade requer a biblioteca 'networkx'. Instale-a com 'pip install networkx' e reinicie a aplicação.")
        else:
            # Importar networkx aqui (já verificamos que está disponível)
            import networkx as nx
            
            # Interface para configurar análise
            col1, col2 = st.columns(2)
            
            with col1:
                min_occurrences = st.slider(
                    "Mínimo de ocorrências:",
                    1, 100, 5,
                    help="Filtrar transições com pelo menos este número de ocorrências"
                )
            
            with col2:
                analysis_scope = st.radio(
                    "Escopo da análise:",
                    ["Por Sessão", "Por Usuário"],
                    index=0,
                    help="Analisar transições dentro da mesma sessão ou para o mesmo usuário entre sessões"
                )
            
            # Executar análise de fluxo
            if st.button("Analisar Fluxo de Eventos"):
                # Contar ocorrências de cada evento
                event_counts = journey_df['event_name'].value_counts()
                
                # Filtrar eventos que atendem ao mínimo de ocorrências
                events_to_include = event_counts[event_counts >= min_occurrences].index.tolist()
                
                if len(events_to_include) >= 2:
                    # Filtrar DataFrame para esses eventos
                    flow_df = journey_df[journey_df['event_name'].isin(events_to_include)].copy()
                    
                    # Determinar agrupamento com base no escopo
                    if analysis_scope == "Por Sessão":
                        groupby_cols = ['client_id', 'session_number']
                    else:
                        groupby_cols = ['client_id']
                    
                    # Ordenar eventos por timestamp
                    flow_df = flow_df.sort_values(groupby_cols + ['event_timestamp'])
                    
                    # Criar grafo dirigido
                    G = nx.DiGraph()
                    
                    # Adicionar nós para cada evento (com contagem)
                    for event, count in event_counts[events_to_include].items():
                        G.add_node(event, count=count)
                    
                    # Construir as arestas
                    edge_counts = {}
                    user_sets = {}  # Para rastrear quais usuários fizeram cada transição
                    
                    # Iterar sobre grupos (sessões ou usuários)
                    for group_key, group_df in flow_df.groupby(groupby_cols):
                        # Obter sequência de eventos
                        events = group_df['event_name'].tolist()
                        client_id = group_key[0]  # O client_id sempre é o primeiro elemento
                        
                        # Criar pares de eventos consecutivos
                        for i in range(len(events) - 1):
                            edge = (events[i], events[i+1])
                            
                            # Incrementar contagem
                            if edge in edge_counts:
                                edge_counts[edge] += 1
                                user_sets[edge].add(client_id)
                            else:
                                edge_counts[edge] = 1
                                user_sets[edge] = {client_id}
                    
                    # Adicionar arestas ao grafo
                    for (source, target), count in edge_counts.items():
                        if count >= min_occurrences:
                            G.add_edge(source, target, 
                                      weight=count, 
                                      users=user_sets[(source, target)])
                    
                    # Verificar se há arestas suficientes
                    if len(G.edges()) > 0:
                        # Calcular métricas de centralidade
                        centrality = nx.betweenness_centrality(G, weight='weight')
                        in_degree = dict(G.in_degree(weight='weight'))
                        out_degree = dict(G.out_degree(weight='weight'))
                        
                        # Adicionar métricas aos nós
                        for node in G.nodes():
                            G.nodes[node]['centrality'] = centrality[node]
                            G.nodes[node]['in_degree'] = in_degree[node]
                            G.nodes[node]['out_degree'] = out_degree[node]
                        
                        # Criar diagrama de Sankey
                        sources = []
                        targets = []
                        values = []
                        labels = []
                        
                        # Coletar nós únicos
                        nodes = set()
                        for u, v in G.edges():
                            nodes.add(u)
                            nodes.add(v)
                        
                        # Mapear nós para índices
                        node_indices = {node: i for i, node in enumerate(nodes)}
                        
                        # Preparar dados para Sankey
                        for u, v, data in G.edges(data=True):
                            sources.append(node_indices[u])
                            targets.append(node_indices[v])
                            values.append(data['weight'])
                        
                        # Lista de labels na ordem correta
                        for node in nodes:
                            labels.append(node)
                        
                        # Criar gráfico Sankey
                        fig = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=labels
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                customdata=[len(G[labels[s]][labels[t]]['users']) 
                                          for s, t in zip(sources, targets)],
                                hovertemplate='%{source.label} → %{target.label}<br>'
                                            'Transições: %{value}<br>'
                                            'Usuários únicos: %{customdata}<extra></extra>'
                            )
                        )])
                        
                        fig.update_layout(
                            title="Fluxo entre Eventos",
                            height=600,
                            font=dict(size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Eventos mais relevantes (baseado na centralidade)
                        st.subheader("Eventos Críticos (Centralidade)")
                        centrality_df = pd.DataFrame({
                            'Evento': list(centrality.keys()),
                            'Centralidade': list(centrality.values()),
                            'Entrada': [in_degree[node] for node in centrality.keys()],
                            'Saída': [out_degree[node] for node in centrality.keys()]
                        }).sort_values('Centralidade', ascending=False)
                        
                        st.dataframe(centrality_df)
                        
                        # Análise de drop-off
                        st.subheader("Análise de Drop-off")
                        
                        # Calcular taxas de drop-off
                        dropoff_data = []
                        
                        for node in G.nodes():
                            in_flow = in_degree.get(node, 0)
                            out_flow = out_degree.get(node, 0)
                            
                            # Calcular drop-off apenas para nós que têm entrada
                            if in_flow > 0:
                                dropoff_rate = (in_flow - out_flow) / in_flow
                                dropoff_data.append({
                                    'Evento': node,
                                    'Entrada': in_flow,
                                    'Saída': out_flow,
                                    'Taxa de Drop-off (%)': round(max(0, dropoff_rate) * 100, 1)
                                })
                        
                        # Criar DataFrame
                        dropoff_df = pd.DataFrame(dropoff_data)
                        dropoff_df = dropoff_df.sort_values('Taxa de Drop-off (%)', ascending=False)
                        
                        # Mostrar tabela
                        st.dataframe(dropoff_df)
                        
                        # Gráfico para top eventos com maior drop-off
                        top_dropoff = dropoff_df.head(10)
                        
                        if not top_dropoff.empty:
                            fig = px.bar(
                                top_dropoff,
                                x='Taxa de Drop-off (%)',
                                y='Evento',
                                orientation='h',
                                labels={
                                    'Taxa de Drop-off (%)': 'Taxa de Drop-off (%)',
                                    'Evento': 'Evento'
                                },
                                title="Top Eventos com Maior Taxa de Drop-off"
                            )
                            
                            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Exportar dados
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Exportar dados de fluxo")
                            
                            # Preparar dados de arestas para exportação
                            edge_data = []
                            for u, v, data in G.edges(data=True):
                                edge_data.append({
                                    'Origem': u,
                                    'Destino': v,
                                    'Transições': data['weight'],
                                    'Usuários Únicos': len(data['users'])
                                })
                            
                            edge_df = pd.DataFrame(edge_data)
                            
                            st.markdown(get_csv_download_link(edge_df, 
                                                      "fluxo_entre_eventos.csv",
                                                      "📥 Baixar dados de fluxo"), 
                                     unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("##### Exportar dados de drop-off")
                            st.markdown(get_csv_download_link(dropoff_df, 
                                                      "analise_dropoff.csv",
                                                      "📥 Baixar dados de drop-off"), 
                                     unsafe_allow_html=True)
                    else:
                        st.warning("Não foram encontradas transições suficientes para análise. Tente reduzir o mínimo de ocorrências.")
                else:
                    st.warning(f"Não há eventos suficientes com pelo menos {min_occurrences} ocorrências. Tente reduzir o valor mínimo.")

# Função principal para executar o dashboard
def main():
    # Interface para upload/seleção do arquivo
    st.sidebar.header("📂 Dados")
    file_option = st.sidebar.radio("Escolha como carregar os dados:", ["Upload de arquivo"])
    
    df = None
    
    # Carregamento dos dados
    if file_option == "Upload de arquivo":
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file=uploaded_file)
    else:
        file_path = st.sidebar.text_input("Digite o caminho completo do arquivo CSV:")
        if file_path:
            df = load_data(file_path=file_path)
    
    # Verificar se os dados foram carregados
    dados_carregados = df is not None
    
    # Seleção da página
    st.sidebar.header("📑 Navegação")
    # Desabilitar páginas de análise se não houver dados
    paginas_disponiveis = ["📌 Início"]
    if dados_carregados:
        paginas_disponiveis += ["👤 Análise por Usuário", "📊 Análise de Eventos", "📈 Jornada do Usuário"]
    
    page = st.sidebar.selectbox(
        "Selecione uma página:",
        paginas_disponiveis
    )
    
    # Página de início (ou se não tiver dados carregados)
    if page == "📌 Início" or not dados_carregados:
        show_home_page()
        if not dados_carregados:
            return
    
    # Continuar apenas se os dados forem carregados
    # Filtros de data
    st.sidebar.header("📅 Filtros")
    # Obter intervalo de datas
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Filtrar por período:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Aplicar filtro de data se necessário
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Ajustar end_date para incluir todo o dia
        end_date = datetime.combine(end_date, datetime.max.time()).date()
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        filtered_df = df
    
    # Mostrar quantos registros foram filtrados
    st.sidebar.info(f"Mostrando {len(filtered_df):,} de {len(df):,} eventos ({(len(filtered_df)/len(df)*100):.1f}%)")
    
    # Exibir a página selecionada
    if page == "👤 Análise por Usuário":
        show_user_analysis_page(filtered_df)
    elif page == "📊 Análise de Eventos":
        show_event_analysis_page(filtered_df)
    elif page == "📈 Jornada do Usuário":
        show_user_journey_page(filtered_df)

# Executar o aplicativo
if __name__ == "__main__":
    main()