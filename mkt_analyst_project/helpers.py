import pandas as pd
import io

class DataFrameMarkdownHelper:
    """
    Classe utilitária para conversão de DataFrames e arquivos tabulares em Markdown BytesIO,
    pronta para upload ou uso em pipelines (ex: OpenAI File Search, aplicações Streamlit etc).

    Métodos principais:
    - format_df_for_markdown: limita linhas/colunas e arredonda floats antes do markdown.
    - df_to_md_bytesio: converte um DataFrame em arquivo Markdown (BytesIO com nome).
    - csv_to_md_bytesio: converte arquivos CSV carregados em BytesIO Markdown.
    - excel_to_md_bytesio: converte arquivos Excel carregados em BytesIO Markdown.
    - to_bytesio_with_name: lê arquivos já suportados (ex: PDF) para BytesIO com nome.
    """

    def __init__(self, max_rows=200, max_cols=30, float_round=4):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.float_round = float_round

    def format_df_for_markdown(self, df: pd.DataFrame) -> pd.DataFrame:
        # Limite de colunas
        if df.shape[1] > self.max_cols:
            df = df.iloc[:, :self.max_cols].copy()
        # Limite de linhas
        if len(df) > self.max_rows:
            df = df.head(self.max_rows).copy()
        # Arredondamento de floats
        for c in df.select_dtypes(include="number").columns:
            df[c] = df[c].round(self.float_round)
        return df

    def df_to_md_bytesio(self, df: pd.DataFrame, file_basename: str) -> io.BytesIO:
        """
        Converte um DataFrame em Markdown (tabela) e retorna como BytesIO ".md".
        """
        df_md = self.format_df_for_markdown(df)
        md_text = df_md.to_markdown(index=False)
        md_full = f'''
# {file_basename}
_Tabela convertida automaticamente para Markdown._
{md_text}
        '''
        bio = io.BytesIO(md_full.encode("utf-8"))
        bio.name = f"{file_basename}.md"
        bio.seek(0)
        return bio

    def csv_to_md_bytesio(self, uploaded) -> io.BytesIO:
        """
        Lê um arquivo CSV carregado (upload/Streamlit) e converte em Markdown BytesIO.
        """
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
        base = uploaded.name.rsplit(".", 1)[0]
        return self.df_to_md_bytesio(df, base)

    def excel_to_md_bytesio(self, uploaded) -> io.BytesIO:
        """
        Lê um arquivo Excel (xlsx/xls) carregado e converte em Markdown BytesIO.
        """
        uploaded.seek(0)
        df = pd.read_excel(uploaded)
        base = uploaded.name.rsplit(".", 1)[0]
        return self.df_to_md_bytesio(df, base)

    def to_bytesio_with_name(self, uploaded) -> io.BytesIO:
        """
        Para arquivos já em formato suportado (ex: PDF), retorna BytesIO nomeado.
        """
        data = uploaded.read()
        bio = io.BytesIO(data)
        bio.name = uploaded.name
        bio.seek(0)
        return bio