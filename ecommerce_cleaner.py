import pandas as pd
import numpy as np

def limpa_valor_compra(valor) -> float:
    """Converte valores com símbolos/vírgulas para float nativo."""
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    
   
    valor_limpo = str(valor).replace("US$", "").replace("$", "").strip()
    valor_limpo = valor_limpo.replace(",", ".")
    return float(valor_limpo)


def padroniza_datas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza signup_time e purchase_time para datetime64[ns]."""
    df = df.copy()
    df['signup_time'] = pd.to_datetime(df['signup_time'], format='mixed')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], format='mixed')
    return df


def cria_feature_velocidade(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a diferença em minutos entre compra e cadastro."""
    df = df.copy()
    diferenca = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 60.0
    
    if (diferenca < 0).any():
        raise ValueError("Anomalia temporal detectada: purchase_time anterior ao signup_time.")
        
    df['velocidade_compra'] = diferenca
    return df


def filtra_idade_comprador(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma idades fora do intervalo [18, 100] em NaN."""
    df = df.copy()
    df['age'] = df['age'].apply(lambda x: x if (18 <= x <= 100) else np.nan)
    return df


def padroniza_navegadores(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa e renomeia variações de nomes de navegadores."""
    df = df.copy()
    df['browser'] = df['browser'].astype(str).str.strip()
    
    mapeamento = {
        'CHROME': 'Chrome',
        'InternetExplorer': 'IE',
        'ie': 'IE'
    }
    
    df['browser'] = df['browser'].replace(mapeamento)
    return df


def remove_transacoes_fantasmas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros com device_id nulo ou string vazia."""
    df = df.copy()

    df['device_id'] = df['device_id'].replace(r'^\s*$', np.nan, regex=True)
    return df.dropna(subset=['device_id'])