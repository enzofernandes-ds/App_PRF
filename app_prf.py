"""
PRF 2025 — O que está associado à gravidade dos acidentes? (Opção A: fatal × não fatal)
=======================================================================================
Aplicação única em Streamlit: importa os dados do Drive, limpa, analisa e modela.

Executar (um único comando, na pasta onde está este arquivo):
    pip install -r requirements.txt
    streamlit run app_prf.py

Organização do arquivo:
    1) Importação e limpeza dos dados
    2) Hipóteses, análise exploratória e modelagem
    3) Interface (abas do Streamlit)
"""

from __future__ import annotations

import inspect
import io
import itertools
import os
import re
import tempfile
import time
import unicodedata
import zipfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder



# ===========================================================================
# PARTE 1 — IMPORTAÇÃO E LIMPEZA DOS DADOS
# ===========================================================================

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
NOME_ZIP = "datatran2025.zip"  # nome fixo da cópia local: só existe UM arquivo desses na pasta de dados
DRIVE_ID = "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR"  # arquivo datatran2025.zip no Drive da PRF
URLS_DOWNLOAD = [
    # endpoint atual do Drive (dispensa a página de "verificação de vírus")
    f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t",
    # endpoint clássico, usado como alternativa
    f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
]


def _pasta_dados() -> Path | None:
    """
    Primeira pasta onde dá para gravar a cópia local do zip: ao lado deste arquivo ou, se o sistema
    negar acesso (ex.: OneDrive/Windows), a pasta temporária. Devolve None se nenhuma servir.
    """
    base = Path(globals().get("__file__", ".")).resolve().parent
    for pasta in (base / "dados", Path(tempfile.gettempdir()) / "prf_dados"):
        try:
            pasta.mkdir(parents=True, exist_ok=True)
            teste = pasta / ".teste_escrita"
            teste.write_bytes(b"ok")
            teste.unlink()
            return pasta
        except OSError:  # PermissionError é um OSError
            continue
    return None

# Valores que a PRF usa para dizer "não sei" e que não são categorias reais
VALORES_AUSENTES = {"ignorado", "não informado", "nao informado", "(null)", "null", "na", "n/a", ""}

# Caixa que contém o território brasileiro (graus decimais)
LAT_MIN, LAT_MAX = -34.0, 5.5
LON_MIN, LON_MAX = -74.0, -34.0


# ---------------------------------------------------------------------------
# 1) IMPORTAÇÃO AUTOMÁTICA
# ---------------------------------------------------------------------------
def _baixar_zip() -> bytes:
    """Tenta cada URL do Drive até receber, de fato, um arquivo .zip."""
    ultimo_erro = "nenhuma URL tentada"
    for url in URLS_DOWNLOAD:
        try:
            resposta = requests.get(url, timeout=120)
            resposta.raise_for_status()
            if zipfile.is_zipfile(io.BytesIO(resposta.content)):
                return resposta.content
            ultimo_erro = "a resposta não é um .zip (provável página de confirmação do Drive)"
        except requests.RequestException as erro:
            ultimo_erro = str(erro)
    raise RuntimeError(f"Falha ao baixar os dados do Drive: {ultimo_erro}")


def _extrair_csv(conteudo: bytes) -> bytes:
    """Abre o .zip em memória e devolve os bytes do (único) CSV de dentro."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError("O zip não contém nenhum arquivo .csv")
        if len(csvs) > 1:
            print(f"[aviso] o zip tem {len(csvs)} CSVs; usando {csvs[0]}")
        with zf.open(csvs[0]) as arquivo:
            return arquivo.read()


def ler_csv(bruto: bytes) -> pd.DataFrame:
    """
    Lê os bytes do CSV da PRF e devolve o DataFrame BRUTO.

    POR QUÊ do encoding: o arquivo de 2025 vem em ISO-8859-1 (latin-1). Tentamos UTF-8 primeiro
    porque a PRF já mudou de codificação entre anos; assim o código não quebra se mudar de novo.
    sep=";" e decimal="," seguem o padrão brasileiro do arquivo (ex.: km "546,2").
    na_values: os textos "NA" (ex.: classificacao_acidente) e "N/A" (regional, delegacia e uop: 2, 22 e 38
    linhas) aparecem no lugar de valor ausente e, sem esse tratamento, virariam categorias falsas.
    """
    parametros = dict(sep=";", decimal=",", na_values=["", "NA", "N/A"], keep_default_na=False, low_memory=False)
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(bruto), encoding=encoding, **parametros)
            print(f"[leitura] {df.shape[0]:,} linhas x {df.shape[1]} colunas (encoding={encoding})")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Não foi possível decodificar o CSV com utf-8 nem latin-1")


def carregar_de_arquivo(conteudo: bytes, nome: str) -> pd.DataFrame:
    """Alternativa manual (ex.: upload no Streamlit): aceita o .zip da PRF ou o .csv já extraído."""
    bruto = _extrair_csv(conteudo) if nome.lower().endswith(".zip") else conteudo
    return ler_csv(bruto)


def _salvar_zip_unico(pasta: Path, conteudo: bytes) -> Path:
    """
    Mantém UM único zip na pasta de dados: grava o novo, apaga o antigo (e cópias numeradas) e põe o novo no lugar.

    O QUE: ao salvar o download, procura arquivos datatran2025*.zip já existentes na pasta (inclusive
           cópias como "datatran2025 (1).zip"), apaga todos e deixa só `datatran2025.zip` com o conteúdo novo.
    POR QUÊ: assim a pasta nunca acumula versões (download, download 1, download 2...) e a base usada é
             sempre a mais recente. A ordem é segura: o arquivo novo é gravado primeiro como `.novo`; se isso
             falhar, o antigo continua intacto e continua servindo de reserva caso o Drive esteja fora do ar.
    """
    destino = pasta / NOME_ZIP
    novo = pasta / (NOME_ZIP + ".novo")
    novo.write_bytes(conteudo)
    antigos = sorted(pasta.glob("datatran2025*.zip"))
    if not antigos:
        print(f"[download] nenhum arquivo anterior em {pasta}: primeira cópia")
    for antigo in antigos:
        try:
            info = antigo.stat()
            antigo.unlink()
            print(f"[download] arquivo antigo removido: {antigo.name} ({info.st_size / 1e6:.1f} MB, "
                  f"de {datetime.fromtimestamp(info.st_mtime):%d/%m/%Y %H:%M})")
        except OSError as erro:
            print(f"[download] não foi possível remover {antigo.name}: {erro}")
    novo.replace(destino)
    return destino


def carregar_dados(forcar_download: bool = True) -> pd.DataFrame:
    """
    Baixa o zip do Drive, extrai o CSV em memória e devolve o DataFrame BRUTO.

    O QUE: a cada execução baixa o .zip, substitui a cópia local (apaga a antiga e grava a nova, sempre com
           o mesmo nome) e lê o único CSV de dentro. Se o download falhar e já existir cópia local, usa a
           cópia e avisa. Se não houver pasta onde gravar, segue só em memória (a cópia é um bônus).
    POR QUÊ: o enunciado exige a base entregue junto do código; rodar o script sempre parte
             da fonte oficial garante que qualquer integrante do grupo reproduza o mesmo
             resultado sem depender de arquivo enviado por chat. A cópia local evita travar
             a apresentação caso o Drive esteja fora do ar ou limite o acesso.
    """
    pasta = _pasta_dados()
    zip_local = pasta / NOME_ZIP if pasta else None
    tem_copia = bool(zip_local and zip_local.exists())

    conteudo = None
    if forcar_download or not tem_copia:
        try:
            conteudo = _baixar_zip()
            if zip_local:
                try:
                    _salvar_zip_unico(pasta, conteudo)
                    print(f"[download] OK — {len(conteudo) / 1e6:.1f} MB salvos em {zip_local}")
                except OSError as erro:
                    print(f"[download] OK, mas a cópia local não pôde ser gravada ({erro}); seguindo em memória")
            else:
                print(f"[download] OK — {len(conteudo) / 1e6:.1f} MB (sem pasta gravável; seguindo em memória)")
        except RuntimeError as erro:
            if not tem_copia:
                raise
            print(f"[download] {erro}\n[download] usando a cópia local {zip_local}")
    if conteudo is None:
        conteudo = zip_local.read_bytes()

    return ler_csv(_extrair_csv(conteudo))


# ---------------------------------------------------------------------------
# 2) LIMPEZA
# ---------------------------------------------------------------------------
def _slug(texto: str) -> str:
    """'Interseção de Vias' -> 'intersecao_de_vias'."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


def _colunas_texto(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]


def limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Aplica a primeira limpeza. Não altera o DataFrame recebido."""
    df = bruto.copy()
    n_inicial = len(df)

    # ------------------------------------------------------------------
    # BLOCO 1 — Nomes de colunas e espaços em texto
    # O QUE: renomeia 'condicao_metereologica' para 'condicao_meteorologica' e
    #        'data_inversa' para 'data'; remove espaços nas pontas de todas as colunas de texto.
    # POR QUÊ: o nome original tem erro de grafia (vira armadilha de digitação no código e
    #          nos slides) e 'data_inversa' é só o formato AAAA-MM-DD, não um conceito
    #          diferente. Espaços invisíveis fariam "Chuva " e "Chuva" virarem categorias distintas.
    # ------------------------------------------------------------------
    df = df.rename(columns={"condicao_metereologica": "condicao_meteorologica", "data_inversa": "data"})
    for col in _colunas_texto(df):
        df[col] = df[col].str.strip()

    # ------------------------------------------------------------------
    # BLOCO 2 — Registros duplicados
    # O QUE: remove linhas com 'id' repetido (mantém a primeira).
    # POR QUÊ: 'id' identifica a ocorrência. A PRF corrige arquivos já publicados, então
    #          duplicatas podem aparecer em versões futuras. Em 2025 há 0 duplicatas; o
    #          bloco é uma proteção para a base não contar o mesmo acidente duas vezes.
    # ------------------------------------------------------------------
    df = df.drop_duplicates(subset="id", keep="first")
    print(f"[limpeza] duplicatas removidas: {n_inicial - len(df)}")

    # ------------------------------------------------------------------
    # BLOCO 3 — "Ignorado" / "Não Informado" viram nulo
    # O QUE: troca esses textos por NaN nas colunas categóricas (hoje afeta
    #        'condicao_meteorologica': 1.000 linhas, e 'sentido_via': 167 linhas).
    # POR QUÊ: eles significam "o policial não registrou", não uma condição climática ou um
    #          sentido real. Deixar como categoria faria o modelo aprender "Ignorado" como se
    #          fosse um padrão do mundo. Como nulo, o tratamento fica explícito e é decidido
    #          depois: a imputação precisa ser ajustada só no treino (após o split), para não
    #          vazar informação do teste.
    # ------------------------------------------------------------------
    for col in _colunas_texto(df):
        df[col] = df[col].mask(df[col].str.lower().isin(VALORES_AUSENTES))

    # ------------------------------------------------------------------
    # BLOCO 4 — Tipos corretos
    # O QUE: 'data' -> datetime; 'horario' -> 'hora' (0-23); cria 'mes'; 'br' -> texto de
    #        3 dígitos ('116' vira '116', '5' vira '005').
    # POR QUÊ: datas lidas como texto não permitem extrair mês nem ordenar no tempo, e a hora
    #          cheia é o que interessa para analisar risco (o minuto exato gera 1.412 valores
    #          distintos e quase nenhum sinal). 'br' é número só na aparência: BR-381 não é
    #          "maior" que BR-116, é um identificador. Como número, um algoritmo trataria a
    #          ordem das rodovias como se tivesse significado.
    # ------------------------------------------------------------------
    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="coerce")
    df["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.hour
    df["mes"] = df["data"].dt.month
    df["br"] = df["br"].astype(str).str.zfill(3)
    df = df.drop(columns="horario")

    # ------------------------------------------------------------------
    # BLOCO 5 — Coordenadas fora do Brasil
    # O QUE: coloca NaN em latitude/longitude que caiam fora da caixa do território brasileiro.
    # POR QUÊ: coordenadas trocadas ou digitadas errado (ex.: sinal invertido) distorcem
    #          qualquer análise espacial e mapas. Em 2025 todas as 72.529 linhas passam na
    #          verificação (a latitude máxima, 4,46, é Roraima); o bloco protege contra
    #          arquivos de outros anos. Preferimos nulo a apagar a linha: o acidente é real,
    #          só a posição é duvidosa.
    # ------------------------------------------------------------------
    fora = ~df["latitude"].between(LAT_MIN, LAT_MAX) | ~df["longitude"].between(LON_MIN, LON_MAX)
    df.loc[fora, ["latitude", "longitude"]] = np.nan
    print(f"[limpeza] coordenadas fora do Brasil anuladas: {int(fora.sum())}")

    # ------------------------------------------------------------------
    # BLOCO 6 — Classificação do acidente ausente
    # O QUE: preenche 'classificacao_acidente' nulo pela regra
    #        mortos > 0 -> 'Com Vítimas Fatais'; senão feridos > 0 -> 'Com Vítimas Feridas';
    #        senão 'Sem Vítimas'. Antes, confere que a regra acerta 100% das linhas já preenchidas.
    # POR QUÊ: a coluna é candidata natural a variável-alvo, e linhas sem alvo não treinam
    #          nem avaliam modelo. Como a classificação é derivada das contagens (em 2025 a
    #          regra bate em todas as linhas preenchidas), dá para recuperar a linha com
    #          segurança em vez de descartá-la. Descartar perderia dado sem necessidade;
    #          imputar pela moda (a classe mais frequente) ignoraria a evidência da própria linha.
    #          Atenção: se 'classificacao_acidente' for o alvo, 'mortos', 'feridos',
    #          'feridos_leves', 'feridos_graves' e 'ilesos' NÃO podem ser variáveis de entrada
    #          (vazamento de alvo: o modelo "leria" a resposta).
    # ------------------------------------------------------------------
    regra = np.select(
        [df["mortos"] > 0, df["feridos"] > 0],
        ["Com Vítimas Fatais", "Com Vítimas Feridas"],
        default="Sem Vítimas",
    )
    regra = pd.Series(regra, index=df.index)
    preenchidas = df["classificacao_acidente"].notna()
    concordancia = (regra[preenchidas] == df.loc[preenchidas, "classificacao_acidente"]).mean()
    print(f"[limpeza] regra de classificação concorda com {concordancia:.2%} das linhas preenchidas")
    if concordancia >= 0.999:
        n_nulos = int((~preenchidas).sum())
        df["classificacao_acidente"] = df["classificacao_acidente"].fillna(regra)
        print(f"[limpeza] classificação recuperada pela regra: {n_nulos} linha(s)")
    else:
        n_nulos = int((~preenchidas).sum())
        df = df[preenchidas]
        print(f"[limpeza] regra não confiável; {n_nulos} linha(s) sem classificação removida(s)")

    # ------------------------------------------------------------------
    # BLOCO 7 — Traçado da via em colunas 0/1
    # O QUE: 'tracado_via' (ex.: "Reta;Declive") vira uma coluna 0/1 por característica
    #        (tracado_reta, tracado_declive, tracado_curva, tracado_ponte...) e a original é removida.
    # POR QUÊ: a coluna junta várias características num só texto e a ordem muda
    #          ("Reta;Declive" e "Declive;Reta" são a mesma via). Isso gera 605 categorias, a
    #          maioria com poucas linhas. Separando, são só 12 características reais, sem
    #          duplicar categorias por causa da ordem.
    # ------------------------------------------------------------------
    tracado = df["tracado_via"].str.get_dummies(sep=";").astype("int8")
    tracado.columns = [f"tracado_{_slug(c)}" for c in tracado.columns]
    df = pd.concat([df.drop(columns="tracado_via"), tracado], axis=1)

    # ------------------------------------------------------------------
    # BLOCO 8 — Contagem de pessoas inconsistente (sinalizar, não corrigir)
    # O QUE: cria 'contagem_inconsistente' = True quando mortos + feridos_leves +
    #        feridos_graves + ilesos + ignorados é MAIOR que 'pessoas'.
    # POR QUÊ: em 2025 isso ocorre em 3.823 acidentes (~5%), com casos extremos (soma
    #          maior que 'pessoas' por dezenas). Não dá para saber qual campo está errado, então
    #          corrigir inventaria dados. Descartar tiraria 5% da base. A flag mantém tudo e
    #          permite testar depois se os resultados mudam sem essas linhas.
    # ------------------------------------------------------------------
    soma = df[["mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados"]].sum(axis=1)
    df["contagem_inconsistente"] = soma > df["pessoas"]
    print(f"[limpeza] linhas com contagem inconsistente (sinalizadas): {int(df['contagem_inconsistente'].sum())}")

    # ------------------------------------------------------------------
    # BLOCO 9 — Outliers: mantidos (decisão registrada)
    # O QUE: nenhum valor é removido ou limitado em 'pessoas', 'veiculos', 'mortos' e 'km'.
    # POR QUÊ: os extremos (acidentes com 76 pessoas ou 82 veículos, 16 mortos) são
    #          compatíveis com eventos reais: ônibus, engavetamentos grandes. O maior 'km' (1.257,
    #          na BR-364) cabe na extensão da própria rodovia. Remover seria apagar justamente os
    #          acidentes mais graves, que são os mais relevantes para estudar gravidade.
    #          O impacto deles deve ser tratado na modelagem (ex.: modelos baseados em árvore
    #          ou transformação log), e não apagando linhas aqui.
    # ------------------------------------------------------------------

    return df.reset_index(drop=True)


# ===========================================================================
# PARTE 2 — HIPÓTESES, ANÁLISE EXPLORATÓRIA E MODELAGEM
# ===========================================================================

SEED = 42

# ---------------------------------------------------------------------------
# ALVO E COLUNAS
# ---------------------------------------------------------------------------
DEFINICOES_ALVO = {
    "Fatal (mortos > 0)": lambda d: d["mortos"] > 0,
    "Fatal ou ferido grave": lambda d: (d["mortos"] > 0) | (d["feridos_graves"] > 0),
}

# Colunas que já contêm (ou são calculadas a partir de) a resposta: nunca entram como variável de entrada.
COLUNAS_VAZAMENTO = [
    "mortos", "feridos", "feridos_leves", "feridos_graves", "ilesos", "ignorados",
    "classificacao_acidente", "contagem_inconsistente",
]
COLUNAS_DESCARTADAS = ["id", "data", "regional", "delegacia", "uop"]
CAT_BAIXA = ["uf", "tipo_acidente", "fase_dia", "sentido_via", "condicao_meteorologica",
             "tipo_pista", "uso_solo", "dia_semana"]
CAT_ALTA = ["municipio", "causa_acidente", "br"]
NUM_BASE = ["km", "pessoas", "veiculos", "hora", "mes", "latitude", "longitude", "fim_de_semana", "madrugada"]
ORDEM_DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def criar_alvo(df: pd.DataFrame, definicao: str) -> pd.Series:
    return DEFINICOES_ALVO[definicao](df).astype("int8").rename("alvo")


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cria hora/fim de semana/madrugada, separa as colunas de entrada e garante que não há vazamento."""
    X = df.copy()
    X["fim_de_semana"] = X["dia_semana"].isin(["sábado", "domingo"]).astype(int)
    X["madrugada"] = X["hora"].between(0, 5).astype(int)
    tracado = [c for c in X.columns if c.startswith("tracado_")]
    num = NUM_BASE + tracado
    cats = CAT_BAIXA + CAT_ALTA
    X = X[num + cats].copy()
    for c in cats:  # nulo vira categoria própria; não aprende nada com os dados, então não vaza
        X[c] = X[c].astype(object).where(X[c].notna(), "Desconhecido")
    assert not set(COLUNAS_VAZAMENTO) & set(X.columns), "coluna de vazamento entrou nas features"
    return X, num


# ---------------------------------------------------------------------------
# ESTATÍSTICA BÁSICA
# ---------------------------------------------------------------------------
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (c - m) / d, (c + m) / d


def taxas(valores: pd.Series, y: pd.Series, ordem: list | None = None) -> pd.DataFrame:
    """Taxa de acidentes fatais (%) por grupo, com intervalo de confiança de Wilson (95%)."""
    tmp = pd.DataFrame({"grupo": valores.values, "y": y.values}).dropna(subset=["grupo"])
    g = tmp.groupby("grupo", observed=True)["y"].agg(fatais="sum", n="count")
    g["taxa_pct"] = g["fatais"] / g["n"] * 100
    ic = [wilson(k, n) for k, n in zip(g["fatais"], g["n"])]
    g["ic_inf_pct"] = [i[0] * 100 for i in ic]
    g["ic_sup_pct"] = [i[1] * 100 for i in ic]
    if ordem is not None:
        g = g.reindex([o for o in ordem if o in g.index])
    return g.reset_index()


def qui2_dois_grupos(y: pd.Series, a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Risco relativo (taxa do grupo a ÷ taxa do grupo b) e p-valor do qui-quadrado 2x2."""
    ya, yb = y[a], y[b]
    tab = [[ya.sum(), len(ya) - ya.sum()], [yb.sum(), len(yb) - yb.sum()]]
    _, p, _, _ = stats.chi2_contingency(tab)
    rr = ya.mean() / yb.mean() if yb.mean() > 0 else np.nan
    return float(rr), float(p)


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"= {p:.3f}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{x:.1f}%".replace(".", ",")


def _veredito(condicoes: list[bool]) -> str:
    if all(condicoes):
        return "Confirmada"
    if any(condicoes):
        return "Parcialmente confirmada"
    return "Não confirmada"


def _taxa_de(tab: pd.DataFrame, grupo: str) -> float:
    linha = tab.loc[tab["grupo"] == grupo, "taxa_pct"]
    return float(linha.iloc[0]) if len(linha) else np.nan


# ---------------------------------------------------------------------------
# HIPÓTESES (guia do projeto, seção 2)
# ---------------------------------------------------------------------------
def testar_hipoteses(df: pd.DataFrame, y: pd.Series) -> list[dict]:
    res = []

    # H1 — noite / madrugada
    t = taxas(df["fase_dia"], y, ["Pleno dia", "Amanhecer", "Anoitecer", "Plena Noite"])
    rr1, p1 = qui2_dois_grupos(y, df["fase_dia"] == "Plena Noite", df["fase_dia"] == "Pleno dia")
    mad = df["hora"].between(0, 5)
    rr1b, p1b = qui2_dois_grupos(y, mad, ~mad)
    res.append(dict(
        id="H1", enunciado="Acidentes à noite ou de madrugada são proporcionalmente mais graves.",
        tabela=t,
        evidencia=(f"Plena noite: {_pct(_taxa_de(t, 'Plena Noite'))} de acidentes fatais contra {_pct(_taxa_de(t, 'Pleno dia'))} "
                   f"em pleno dia (risco relativo {rr1:.2f}, p {fmt_p(p1)}). Madrugada (0h–5h59): "
                   f"{_pct(y[mad].mean() * 100)} contra {_pct(y[~mad].mean() * 100)} no resto do dia (risco relativo {rr1b:.2f}, p {fmt_p(p1b)})."),
        veredito=_veredito([rr1 > 1 and p1 < 0.01, rr1b > 1 and p1b < 0.01]),
    ))

    # H2 — tipo de acidente
    t = taxas(df["tipo_acidente"], y).sort_values("taxa_pct", ascending=False)
    base = df["tipo_acidente"] == "Colisão traseira"
    rr_f, p_f = qui2_dois_grupos(y, df["tipo_acidente"] == "Colisão frontal", base)
    rr_a, p_a = qui2_dois_grupos(y, df["tipo_acidente"] == "Atropelamento de Pedestre", base)
    res.append(dict(
        id="H2", enunciado="Colisão frontal e atropelamento têm maior letalidade que colisão traseira.",
        tabela=t,
        evidencia=(f"Colisão traseira: {_pct(_taxa_de(t, 'Colisão traseira'))}. Colisão frontal: {_pct(_taxa_de(t, 'Colisão frontal'))} "
                   f"(risco relativo {rr_f:.1f}, p {fmt_p(p_f)}). Atropelamento de pedestre: {_pct(_taxa_de(t, 'Atropelamento de Pedestre'))} "
                   f"(risco relativo {rr_a:.1f}, p {fmt_p(p_a)})."),
        veredito=_veredito([rr_f > 1 and p_f < 0.01, rr_a > 1 and p_a < 0.01]),
    ))

    # H3 — tipo de pista
    t = taxas(df["tipo_pista"], y, ["Simples", "Dupla", "Múltipla"])
    rr3, p3 = qui2_dois_grupos(y, df["tipo_pista"] == "Simples", df["tipo_pista"] == "Dupla")
    res.append(dict(
        id="H3", enunciado="Pista simples é mais letal que pista dupla.",
        tabela=t,
        evidencia=(f"Pista simples: {_pct(_taxa_de(t, 'Simples'))}; dupla: {_pct(_taxa_de(t, 'Dupla'))}; "
                   f"múltipla: {_pct(_taxa_de(t, 'Múltipla'))}. Simples vs. dupla: risco relativo {rr3:.2f}, p {fmt_p(p3)}."),
        veredito=_veredito([rr3 > 1 and p3 < 0.01]),
    ))

    # H4 — fim de semana
    t = taxas(df["dia_semana"], y, ORDEM_DIAS)
    fds = df["dia_semana"].isin(["sábado", "domingo"])
    rr4, p4 = qui2_dois_grupos(y, fds, ~fds)
    res.append(dict(
        id="H4", enunciado="Fins de semana têm mais acidentes graves (álcool e velocidade).",
        tabela=t,
        evidencia=(f"Sábado e domingo concentram {_pct(fds.mean() * 100)} dos acidentes e {_pct(y[fds].sum() / y.sum() * 100)} dos fatais; "
                   f"taxa de fatalidade {_pct(y[fds].mean() * 100)} contra {_pct(y[~fds].mean() * 100)} nos dias úteis "
                   f"(risco relativo {rr4:.2f}, p {fmt_p(p4)})."),
        veredito=_veredito([rr4 > 1 and p4 < 0.01]),
    ))

    # H5 — concentração por UF / BR, independente do volume
    t_uf = taxas(df["uf"], y)
    t_uf = t_uf[t_uf["n"] >= 300].sort_values("taxa_pct", ascending=False)
    ct = pd.crosstab(df["uf"], y)
    _, p5, _, _ = stats.chi2_contingency(ct)
    rho = stats.spearmanr(t_uf["n"], t_uf["taxa_pct"]).correlation
    alto, baixo = t_uf.iloc[0], t_uf.iloc[-1]
    acima = t_uf[t_uf["ic_inf_pct"] > y.mean() * 100]["grupo"].tolist()
    t_br = taxas(df["br"], y)
    t_br = t_br[t_br["n"] >= 300].sort_values("taxa_pct", ascending=False)
    top_br = ", ".join(f"BR-{r.grupo} ({_pct(r.taxa_pct)}, n={int(r.n)})" for r in t_br.head(3).itertuples())
    razao = alto["taxa_pct"] / baixo["taxa_pct"]
    res.append(dict(
        id="H5", enunciado="Alguns estados ou BRs concentram a letalidade, independentemente do volume de acidentes.",
        tabela=t_uf,
        evidencia=(f"Entre UFs com 300+ acidentes, a taxa vai de {_pct(baixo['taxa_pct'])} ({baixo['grupo']}) a {_pct(alto['taxa_pct'])} ({alto['grupo']}), "
                   f"razão de {razao:.1f}× (qui-quadrado entre UFs: p {fmt_p(p5)}). A correlação de Spearman entre volume e taxa é {rho:.2f}. "
                   f"UFs com taxa significativamente acima da média nacional: {', '.join(acima) if acima else 'nenhuma'}. "
                   f"BRs com maior taxa (300+ acidentes): {top_br}."),
        veredito=_veredito([p5 < 0.01 and razao >= 1.5, abs(rho) < 0.6]) if p5 < 0.01 else "Não confirmada",
    ))
    return res


# ---------------------------------------------------------------------------
# ANÁLISE EXPLORATÓRIA (guia do projeto, seção 3)
# ---------------------------------------------------------------------------
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér com correção de viés (Bergsma)."""
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def _faltantes(bruto: pd.DataFrame, limpo: pd.DataFrame) -> pd.DataFrame:
    renomeadas = {"condicao_metereologica": "condicao_meteorologica", "data_inversa": "data", "horario": "hora"}
    linhas = []
    for c in bruto.columns:
        s = bruto[c]
        nan = int(s.isna().sum())
        tokens = int(s[s.notna()].astype(str).str.strip().str.lower().isin(VALORES_AUSENTES).sum())
        destino = renomeadas.get(c, c)
        nulos_depois = int(limpo[destino].isna().sum()) if destino in limpo.columns else None
        if nan + tokens > 0:
            linhas.append(dict(coluna=c, nulos_brutos=nan, ignorado_ou_nao_informado=tokens,
                               total_ausentes=nan + tokens, pct_ausentes=(nan + tokens) / len(bruto) * 100,
                               nulos_apos_limpeza=nulos_depois))
    return pd.DataFrame(linhas).sort_values("total_ausentes", ascending=False).reset_index(drop=True)


def _informatividade_ausencia(limpo: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    linhas = []
    for c in ["condicao_meteorologica", "sentido_via", "regional", "delegacia", "uop"]:
        aus = limpo[c].isna()
        if aus.sum() == 0:
            continue
        if aus.sum() >= 5:
            rr, p = qui2_dois_grupos(y, aus, ~aus)
        else:
            rr, p = np.nan, np.nan
        linhas.append(dict(coluna=c, linhas_ausentes=int(aus.sum()),
                           taxa_fatal_ausente_pct=float(y[aus].mean() * 100),
                           taxa_fatal_preenchido_pct=float(y[~aus].mean() * 100), risco_relativo=rr, p_valor=p))
    return pd.DataFrame(linhas)


def _outliers(limpo: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    linhas, mascara_total = [], pd.Series(False, index=limpo.index)
    for c in ["km", "pessoas", "veiculos"]:
        q1, q3 = limpo[c].quantile([0.25, 0.75])
        lim = q3 + 1.5 * (q3 - q1)
        acima = limpo[c] > lim
        mascara_total |= acima
        linhas.append(dict(variavel=c, mediana=float(limpo[c].median()), p99=float(limpo[c].quantile(0.99)),
                           maximo=float(limpo[c].max()), limite_iqr=float(lim), n_acima=int(acima.sum()),
                           pct_acima=float(acima.mean() * 100),
                           taxa_fatal_acima_pct=float(y[acima].mean() * 100) if acima.any() else np.nan,
                           taxa_fatal_abaixo_pct=float(y[~acima].mean() * 100)))
    top = limpo.sort_values("pessoas", ascending=False).head(10)[
        ["data", "uf", "br", "km", "tipo_acidente", "pessoas", "veiculos", "mortos"]]
    efeito = dict(n_removidas=int(mascara_total.sum()), pct_removidas=float(mascara_total.mean() * 100),
                  fatais_perdidos=int(y[mascara_total].sum()), pct_fatais_perdidos=float(y[mascara_total].sum() / y.sum() * 100))
    return pd.DataFrame(linhas), top, efeito


def _faixas(serie: pd.Series, y: pd.Series, kind: str) -> pd.DataFrame:
    if kind == "km":
        f = pd.qcut(serie, 10, duplicates="drop")
        ordem = [str(c) for c in f.cat.categories]
        return taxas(f.astype(str), y, ordem)
    if kind == "veiculos":
        f = serie.clip(upper=5).map(lambda v: "5+" if v >= 5 else str(int(v)))
        return taxas(f, y, ["1", "2", "3", "4", "5+"])
    rotulos = ["1", "2", "3", "4", "5–6", "7–10", "11+"]
    f = pd.cut(serie, [0, 1, 2, 3, 4, 6, 10, np.inf], labels=rotulos)
    return taxas(f.astype(str), y, rotulos)


def eda(bruto: pd.DataFrame, limpo: pd.DataFrame, y: pd.Series) -> dict:
    out: dict = {}
    out["faltantes"] = _faltantes(bruto, limpo)
    out["ausencia_info"] = _informatividade_ausencia(limpo, y)
    out["balanceamento"] = dict(n=int(len(y)), fatais=int(y.sum()), pct_fatais=float(y.mean() * 100))
    out["outliers"], out["outliers_top"], out["outliers_efeito"] = _outliers(limpo, y)

    cols_num = ["km", "pessoas", "veiculos", "hora", "mes", "latitude", "longitude",
                "mortos", "feridos_leves", "feridos_graves", "feridos", "ilesos", "ignorados"]
    num = limpo[cols_num].assign(alvo=y.values)
    out["corr_pearson"] = num.corr(method="pearson")
    out["corr_spearman"] = num.corr(method="spearman")

    cats = ["uf", "br", "tipo_acidente", "causa_acidente", "fase_dia", "condicao_meteorologica",
            "tipo_pista", "uso_solo", "sentido_via", "dia_semana"]
    cat_df = limpo[cats].copy()
    cat_df["alvo"] = y.map({0: "não fatal", 1: "fatal"}).values
    nomes = list(cat_df.columns)
    mat = pd.DataFrame(np.eye(len(nomes)), index=nomes, columns=nomes)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            mat.loc[a, b] = mat.loc[b, a] = cramers_v(cat_df[a], cat_df[b])
    out["cramers"] = mat

    hora = taxas(limpo["hora"], y).sort_values("grupo")
    lin = {"hora": hora, "km": _faixas(limpo["km"], y, "km"),
           "veiculos": _faixas(limpo["veiculos"], y, "veiculos"), "pessoas": _faixas(limpo["pessoas"], y, "pessoas")}
    out["linearidade"] = lin
    resumo = []
    for nome, t in lin.items():
        rho = stats.spearmanr(range(len(t)), t["taxa_pct"]).correlation
        resumo.append(dict(variavel=nome, rho_monotonia=float(rho), faixa_maior_taxa=str(t.loc[t["taxa_pct"].idxmax(), "grupo"]),
                           faixa_menor_taxa=str(t.loc[t["taxa_pct"].idxmin(), "grupo"]),
                           taxa_max_pct=float(t["taxa_pct"].max()), taxa_min_pct=float(t["taxa_pct"].min())))
    out["linearidade_resumo"] = pd.DataFrame(resumo)

    linhas = []
    for c in CAT_BAIXA + CAT_ALTA + ["tracado_via (original)"]:
        if c == "tracado_via (original)":
            linhas.append(dict(coluna="tracado_via (antes da limpeza)", categorias=int(bruto["tracado_via"].nunique()),
                               pct_maior_categoria=float(bruto["tracado_via"].value_counts(normalize=True).iloc[0] * 100),
                               categorias_com_menos_de_30=int((bruto["tracado_via"].value_counts() < 30).sum())))
            continue
        vc = limpo[c].value_counts()
        linhas.append(dict(coluna=c, categorias=int(len(vc)), pct_maior_categoria=float(vc.iloc[0] / vc.sum() * 100),
                           categorias_com_menos_de_30=int((vc < 30).sum())))
    out["cardinalidade"] = pd.DataFrame(linhas).sort_values("categorias", ascending=False).reset_index(drop=True)
    vc = limpo["municipio"].value_counts()
    raros = vc < 30
    out["municipio_cauda"] = dict(n=int(len(vc)), pct_municipios_raros=float(raros.mean() * 100),
                                  pct_acidentes_em_raros=float(vc[raros].sum() / vc.sum() * 100),
                                  contagens=vc.values)

    out["temporal"] = {
        "hora": hora,
        "dia_semana": taxas(limpo["dia_semana"], y, ORDEM_DIAS),
        "mes": taxas(limpo["mes"], y).sort_values("grupo"),
    }
    out["geo_uf"] = taxas(limpo["uf"], y).sort_values("taxa_pct", ascending=False)
    t_br = taxas(limpo["br"], y)
    out["geo_br"] = t_br[t_br["n"] >= 300].sort_values("taxa_pct", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# MODELAGEM (guia do projeto, seção 5)
# ---------------------------------------------------------------------------
def construir_pipeline(modelo, num_cols: list[str], escalar: bool = False) -> Pipeline:
    etapas_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        etapas_num.append(("esc", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(etapas_num), num_cols),
        ("cat_baixa", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), CAT_BAIXA),
        ("cat_alta", TargetEncoder(target_type="binary", random_state=SEED), CAT_ALTA),
    ])
    return Pipeline([("pre", pre), ("model", modelo)])


def metricas(y_true, proba, limiar: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= limiar).astype(int)
    return dict(
        acuracia=accuracy_score(y_true, pred),
        precisao=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        pr_auc=average_precision_score(y_true, proba),
        roc_auc=roc_auc_score(y_true, proba),
    )


def dividir(df: pd.DataFrame, definicao: str):
    y = criar_alvo(df, definicao)
    X, num = preparar_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    return X_tr, X_te, y_tr, y_te, num


GRADE_BUSCA = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0],
    "model__max_iter": [150, 300],
}


def _paralelo(funcao, n_jobs: int):
    """Roda funcao(n_jobs). Se o paralelismo falhar (ex.: restrição do sistema), repete em série."""
    try:
        return funcao(n_jobs)
    except Exception:
        if n_jobs == 1:
            raise
        return funcao(1)


def treinar_tudo(df: pd.DataFrame, definicao: str, usar_busca: bool = True, n_iter: int = 8,
                 rapido: bool = True, progresso=None) -> dict:
    """
    Treina todos os modelos. `rapido=True` troca um pouco de rigor por velocidade:
    busca de hiperparâmetros em amostra estratificada de 40% do treino (e reajuste do melhor no treino completo),
    até 6 combinações, Random Forest com 100 árvores e importância por permutação em 3.000 acidentes.
    Em qualquer modo, busca, validação cruzada e importância usam todos os núcleos da CPU.
    `progresso(fração, texto)` é chamado a cada etapa, para a interface mostrar uma barra de progresso.
    """
    inicio = time.perf_counter()
    nucleos = os.cpu_count() or 1
    n_jobs = -1 if nucleos > 1 else 1
    n_arvores, n_importancia, repeticoes = (100, 3000, 2) if rapido else (200, 5000, 3)
    n_iter_efetivo = min(n_iter, 6) if rapido else n_iter
    frac_busca = 0.4 if rapido else 1.0

    X_tr, X_te, y_tr, y_te, num = dividir(df, definicao)
    hgb = lambda **kw: HistGradientBoostingClassifier(random_state=SEED, **kw)  # noqa: E731
    modelos = {
        "Dummy (sempre 'não fatal')": construir_pipeline(DummyClassifier(strategy="most_frequent"), num),
        "Regressão Logística (baseline)": construir_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"), num, escalar=True),
        "Random Forest": construir_pipeline(
            RandomForestClassifier(n_estimators=n_arvores, min_samples_leaf=3, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED), num),
        "HistGradientBoosting (sem balanceamento)": construir_pipeline(hgb(), num),
        "HistGradientBoosting (class_weight)": construir_pipeline(hgb(class_weight="balanced"), num),
    }
    total, feitos = len(modelos) + (1 if usar_busca else 0) + 2, 0

    def avancar(texto: str) -> None:
        if progresso:
            progresso(feitos / total, texto)

    for nome, p in modelos.items():
        avancar(f"Treinando: {nome}")
        p.fit(X_tr, y_tr)
        feitos += 1

    busca_tab, melhores = None, None
    principal_nome = "HistGradientBoosting (class_weight)"
    if usar_busca:
        avancar(f"Busca de hiperparâmetros ({n_iter_efetivo} combinações × 3 dobras"
                + (", em amostra de 40% do treino)" if rapido else ")"))
        base = construir_pipeline(hgb(class_weight="balanced"), num)
        if frac_busca < 1:
            X_b, _, y_b, _ = train_test_split(X_tr, y_tr, train_size=frac_busca, stratify=y_tr, random_state=SEED)
        else:
            X_b, y_b = X_tr, y_tr
        busca = RandomizedSearchCV(base, GRADE_BUSCA, n_iter=n_iter_efetivo, scoring="average_precision",
                                   cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED,
                                   n_jobs=n_jobs, refit=False)
        _paralelo(lambda nj: busca.set_params(n_jobs=nj).fit(X_b, y_b), n_jobs)
        principal_nome = "HistGradientBoosting (ajustado por busca)"
        modelos[principal_nome] = clone(base).set_params(**busca.best_params_).fit(X_tr, y_tr)  # reajuste no treino completo
        melhores = {k.replace("model__", ""): v for k, v in busca.best_params_.items()}
        busca_tab = (pd.DataFrame(busca.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
                     .sort_values("rank_test_score").reset_index(drop=True))
        busca_tab["params"] = busca_tab["params"].map(lambda d: ", ".join(f"{k.replace('model__', '')}={v}" for k, v in d.items()))
        busca_tab = busca_tab.rename(columns={"mean_test_score": "PR-AUC médio (CV)", "std_test_score": "desvio",
                                              "rank_test_score": "posto"})
        feitos += 1

    probas = {nome: m.predict_proba(X_te)[:, 1] for nome, m in modelos.items()}
    comparacao = pd.DataFrame({n: metricas(y_te, p) for n, p in probas.items()}).T

    # limiar escolhido por validação cruzada no TREINO (não no teste)
    avancar("Escolhendo o limiar de decisão por validação cruzada")
    oof = _paralelo(lambda nj: cross_val_predict(
        clone(modelos[principal_nome]), X_tr, y_tr, method="predict_proba",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=nj)[:, 1], n_jobs)
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    limiar_cv = float(thr[int(np.argmax(f1s))])
    feitos += 1

    avancar("Calculando a importância das variáveis")
    amostra = X_te.sample(min(n_importancia, len(X_te)), random_state=SEED)
    imp = _paralelo(lambda nj: permutation_importance(
        modelos[principal_nome], amostra, y_te.loc[amostra.index], scoring="average_precision",
        n_repeats=repeticoes, random_state=SEED, n_jobs=nj), n_jobs)
    importancias = (pd.DataFrame({"variavel": amostra.columns, "queda_pr_auc": imp.importances_mean,
                                  "desvio": imp.importances_std})
                    .sort_values("queda_pr_auc", ascending=False).reset_index(drop=True))
    feitos += 1
    avancar("Concluído")

    return dict(
        y_te=y_te.values, probas=probas, comparacao=comparacao, principal=principal_nome, limiar_cv=limiar_cv,
        busca_tab=busca_tab, melhores_params=melhores, importancias=importancias,
        split=dict(n_treino=len(y_tr), n_teste=len(y_te), pct_fatal_treino=float(y_tr.mean() * 100),
                   pct_fatal_teste=float(y_te.mean() * 100)),
        features=dict(numericas=num, categoricas_baixa=CAT_BAIXA, categoricas_alta=CAT_ALTA),
        tempo_s=time.perf_counter() - inicio, modo="rápido" if rapido else "completo", nucleos=nucleos,
    )


def matriz_confusao(y_true, proba, limiar: float) -> np.ndarray:
    return confusion_matrix(y_true, (np.asarray(proba) >= limiar).astype(int), labels=[0, 1])


def curva_pr(y_true, proba) -> tuple[np.ndarray, np.ndarray]:
    p, r, _ = precision_recall_curve(y_true, proba)
    return r, p


# ---------------------------------------------------------------------------
# TENTATIVAS QUE DERAM ERRADO (demonstrações)
# ---------------------------------------------------------------------------
def _oversample_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    return np.concatenate([np.arange(len(y)), extra])


def demo_oversampling(df: pd.DataFrame, definicao: str, n_max: int | None = None, n_arvores: int = 100) -> pd.DataFrame:
    """Oversampling ANTES do split (errado) × DEPOIS do split, só no treino (certo)."""
    y = criar_alvo(df, definicao)
    X, num = preparar_features(df)
    if n_max and len(X) > n_max:
        X, _, y, _ = train_test_split(X, y, train_size=n_max, stratify=y, random_state=SEED)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    novo = lambda: construir_pipeline(RandomForestClassifier(  # noqa: E731
        n_estimators=n_arvores, min_samples_leaf=1, n_jobs=-1, random_state=SEED), num)
    rng = np.random.default_rng(SEED)

    idx = _oversample_idx(y.values, rng)  # ERRADO: duplica antes de separar teste
    Xo, yo = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    a, b, ya, yb = train_test_split(Xo, yo, test_size=0.2, stratify=yo, random_state=SEED)
    r_errado = metricas(yb, novo().fit(a, ya).predict_proba(b)[:, 1])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)  # CERTO
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    idx = _oversample_idx(y_tr.values, rng)
    r_certo = metricas(y_te, novo().fit(X_tr.iloc[idx], y_tr.iloc[idx]).predict_proba(X_te)[:, 1])
    return pd.DataFrame({"Oversampling ANTES do split (errado)": r_errado,
                         "Oversampling DEPOIS do split, só no treino (certo)": r_certo}).T


def demo_vazamento(df: pd.DataFrame, definicao: str) -> pd.DataFrame:
    """Modelo com e sem as colunas que contêm a resposta (feridos, ilesos, ignorados)."""
    y = criar_alvo(df, definicao)
    X, num = preparar_features(df)
    vaza = ["feridos", "feridos_leves", "feridos_graves", "ilesos", "ignorados"]
    X2 = X.copy()
    for c in vaza:
        X2[c] = df[c].values
    out = {}
    for nome, (Xm, nm) in {"Sem vazamento (features legítimas)": (X, num),
                           "Com vazamento (+ feridos, ilesos, ignorados)": (X2, num + vaza)}.items():
        a, b, ya, yb = train_test_split(Xm, y, test_size=0.2, stratify=y, random_state=SEED)
        m = construir_pipeline(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), nm).fit(a, ya)
        out[nome] = metricas(yb, m.predict_proba(b)[:, 1])
    return pd.DataFrame(out).T


# ===========================================================================
# PARTE 3 — INTERFACE (STREAMLIT)
# ===========================================================================

st.set_page_config(page_title="PRF 2025 — Gravidade dos acidentes", page_icon="🚧", layout="wide")

_contador = itertools.count()
_W_DF = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}
_W_PLOT = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}


def mostrar_df(df, **kw):
    st.dataframe(df, **_W_DF, **kw)


def mostrar_plot(fig):
    st.plotly_chart(fig, key=f"plot_{next(_contador)}", **_W_PLOT)


def br(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct_br(x: float, d: int = 1) -> str:
    return f"{br(x, d)}%"


def mil(n) -> str:
    """Milhar com ponto: 72529 -> 72.529"""
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# DADOS (cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Baixando e limpando os dados da PRF…")
def carregar_base(arquivo_bytes: bytes | None, nome: str | None):
    log = io.StringIO()
    with redirect_stdout(log):
        bruto = carregar_de_arquivo(arquivo_bytes, nome) if arquivo_bytes else carregar_dados()
        limpo = limpar(bruto)
    return bruto, limpo, log.getvalue()


@st.cache_data(show_spinner="Calculando hipóteses e análise exploratória…")
def calcular_analises(_bruto, _limpo, chave, alvo_nome):
    y = criar_alvo(_limpo, alvo_nome)
    return testar_hipoteses(_limpo, y), eda(_bruto, _limpo, y)



@st.cache_resource(show_spinner=False)
def memoria_modelos() -> dict:
    """Guarda os modelos já treinados (um por configuração), para não treinar de novo à toa."""
    return {}


@st.cache_resource(show_spinner=False)
def rodar_oversampling(_limpo, chave, alvo_nome, amostra):
    return demo_oversampling(_limpo, alvo_nome, n_max=30000 if amostra else None)


@st.cache_resource(show_spinner=False)
def rodar_vazamento(_limpo, chave, alvo_nome):
    return demo_vazamento(_limpo, alvo_nome)


# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------
st.sidebar.title("🚧 PRF 2025")
st.sidebar.caption("Acidentes em rodovias federais — dados abertos da PRF (agrupados por ocorrência).")
alvo_nome = st.sidebar.selectbox("Definição do alvo (gravidade)", list(DEFINICOES_ALVO), index=0)
with st.sidebar.expander("Fonte dos dados"):
    st.caption("A cada execução o app baixa o zip do Drive. Se o download falhar, envie o zip ou o CSV manualmente.")
    arquivo = st.file_uploader("datatran2025 (.zip ou .csv)", type=["zip", "csv"])
    if st.button("Baixar novamente do Drive"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

try:
    bruto, limpo, log_carga = carregar_base(arquivo.getvalue() if arquivo else None, arquivo.name if arquivo else None)
except RuntimeError as erro:
    st.error(f"Não foi possível carregar os dados: {erro}\n\nEnvie o arquivo datatran2025 (.zip ou .csv) na barra lateral.")
    st.stop()

chave = (len(limpo), int(limpo["id"].sum()))
y = criar_alvo(limpo, alvo_nome)
hipoteses, E = calcular_analises(bruto, limpo, chave, alvo_nome)
geral = float(y.mean() * 100)

st.title("Gravidade dos acidentes nas rodovias federais — PRF 2025")
st.caption(f"{mil(len(limpo))} acidentes · alvo atual: **{alvo_nome}** ({pct_br(geral)} dos acidentes)")

tabs = st.tabs(["1 · Objetivo e hipóteses", "2 · Análise exploratória", "3 · Modelagem", "4 · O que deu errado"])


# ===========================================================================
# ABA 1 — OBJETIVO E HIPÓTESES
# ===========================================================================
def barra_taxas(tab: pd.DataFrame, altura: int = 340):
    fig = go.Figure(go.Bar(
        x=tab["grupo"].astype(str), y=tab["taxa_pct"],
        error_y=dict(type="data", symmetric=False, array=(tab["ic_sup_pct"] - tab["taxa_pct"]).tolist(),
                     arrayminus=(tab["taxa_pct"] - tab["ic_inf_pct"]).tolist()),
        customdata=np.stack([tab["n"], tab["fatais"]], axis=-1),
        hovertemplate="%{x}<br>taxa: %{y:.1f}%<br>acidentes: %{customdata[0]}<br>com alvo=1: %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=geral, line_dash="dash", annotation_text=f"média geral {geral:.1f}%".replace(".", ","))
    fig.update_layout(yaxis_title="% de acidentes graves", xaxis=dict(type="category"), height=altura,
                      margin=dict(t=30, b=10))
    return fig


def linha_taxas(tab: pd.DataFrame, titulo: str, altura: int = 300):
    fig = px.line(tab, x=tab["grupo"].astype(str), y="taxa_pct", markers=True, title=titulo,
                  labels={"x": "", "taxa_pct": "% de acidentes graves"})
    fig.add_hline(y=geral, line_dash="dash")
    fig.update_layout(height=altura, margin=dict(t=40, b=10), xaxis=dict(type="category"))
    return fig


with tabs[0]:
    st.header("Objetivo, alvo e hipóteses")
    st.markdown(
        "**Pergunta:** com os padrões de local, horário, clima, tipo de pista e tipo de acidente, **quais combinações elevam o "
        "risco de morte** nas rodovias federais? A resposta ajuda a priorizar fiscalização e sinalização.\n\n"
        f"**Alvo (classificação binária):** `{alvo_nome}`. Fatais são minoria: **{pct_br(geral)}** dos acidentes "
        f"({mil(y.sum())} de {mil(len(y))}), desbalanceamento natural que a rubrica pede para tratar.")
    st.warning("**Vazamento de dados:** `mortos`, `feridos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, "
               "`classificacao_acidente` e a flag `contagem_inconsistente` (calculada a partir deles) **não entram como "
               "variáveis de entrada**: já contêm a resposta.")

    st.subheader("Hipóteses iniciais × o que os dados mostraram")
    mostrar_df(pd.DataFrame([{"": h["id"], "Hipótese": h["enunciado"], "Resultado": h["veredito"]} for h in hipoteses]),
               hide_index=True)
    for h in hipoteses:
        with st.container(border=True):
            st.markdown(f"**{h['id']} — {h['enunciado']}**")
            {"Confirmada": st.success, "Parcialmente confirmada": st.warning, "Não confirmada": st.error}[h["veredito"]](h["veredito"])
            st.markdown(h["evidencia"])
            mostrar_plot(barra_taxas(h["tabela"]))
    st.caption("Barras de erro: intervalo de confiança de 95% (Wilson). Teste de significância: qui-quadrado; limiar de 1%. "
               "Associação não é causalidade: por exemplo, a taxa maior à noite pode refletir visibilidade, velocidade ou álcool.")


# ===========================================================================
# ABA 2 — ANÁLISE EXPLORATÓRIA
# ===========================================================================
def decisao(texto: str):
    st.info(f"**Decisão gerada:** {texto}")


with tabs[1]:
    st.header("Análise exploratória → decisões")
    st.caption("Cada achado termina em uma decisão (tratamento, algoritmo ou métrica).")
    sub = st.tabs(["Faltantes", "Balanceamento", "Outliers", "Correlação", "Linearidade", "Cardinalidade", "Temporal", "Geográfica"])

    with sub[0]:
        st.subheader("Valores faltantes")
        mostrar_df(E["faltantes"].round(3), hide_index=True)
        st.markdown("**A ausência diz algo sobre o alvo?** Taxa de acidentes graves quando o campo está ausente × preenchido:")
        mostrar_df(E["ausencia_info"].round(3), hide_index=True)
        aus = E["ausencia_info"].set_index("coluna")
        cm = aus.loc["condicao_meteorologica"]
        sv = aus.loc["sentido_via"]
        decisao(
            f"`condicao_meteorologica` ausente: {pct_br(cm['taxa_fatal_ausente_pct'])} de graves contra {pct_br(cm['taxa_fatal_preenchido_pct'])} "
            f"quando preenchida (p {fmt_p(cm['p_valor'])}) — a ausência {'é' if cm['p_valor'] < 0.05 else 'não é'} informativa; entra como categoria própria "
            "«Desconhecido» (alternativa descartada: imputar a moda «Céu Claro», que inventaria um clima). "
            f"`sentido_via` ausente: {pct_br(sv['taxa_fatal_ausente_pct'])} de graves em {int(sv['linhas_ausentes'])} linhas; mesmo tratamento. "
            f"`regional`, `delegacia` e `uop` (poucos nulos, {limpo['uop'].nunique()} categorias em `uop`, redundantes com `uf`) saem das variáveis de entrada "
            "(alternativa descartada: imputar, o que seria esforço sem ganho). Imputação numérica (mediana) é ajustada só no treino.")

    with sub[1]:
        st.subheader("Balanceamento de classes")
        bal = E["balanceamento"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Acidentes", f"{bal['n']:,}".replace(",", "."))
        c2.metric("Graves (alvo = 1)", f"{bal['fatais']:,}".replace(",", "."))
        c3.metric("% graves", pct_br(bal["pct_fatais"]))
        fig = px.bar(x=["Não grave", "Grave"], y=[bal["n"] - bal["fatais"], bal["fatais"]], labels={"x": "", "y": "acidentes"})
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        mostrar_plot(fig)
        decisao(
            f"um modelo que sempre responde «não grave» acerta {pct_br(100 - bal['pct_fatais'])} e detecta zero casos graves. Por isso a métrica "
            "principal é **recall/F1 da classe grave e PR-AUC**, não acurácia. Balanceamento via `class_weight` dentro do treino "
            "(alternativa descartada: oversampling por duplicação de linhas ou SMOTE; se usado, só depois da divisão treino/teste — antes dela o resultado infla, veja a aba 4). "
            "A divisão treino/teste é estratificada para manter a proporção.")

    with sub[2]:
        st.subheader("Outliers")
        mostrar_df(E["outliers"].round(2), hide_index=True)
        c1, c2 = st.columns(2)
        f1 = px.box(limpo, y="pessoas", log_y=True, points="outliers", title="Pessoas por acidente (escala log)")
        f1.update_layout(height=320, margin=dict(t=40, b=10))
        c1.plotly_chart(f1, key=f"plot_{next(_contador)}", **_W_PLOT)
        f2 = px.box(limpo, y="veiculos", log_y=True, points="outliers", title="Veículos por acidente (escala log)")
        f2.update_layout(height=320, margin=dict(t=40, b=10))
        c2.plotly_chart(f2, key=f"plot_{next(_contador)}", **_W_PLOT)
        st.caption("Os 10 maiores acidentes por número de pessoas (ônibus, engavetamentos):")
        mostrar_df(E["outliers_top"], hide_index=True)
        ef, op = E["outliers_efeito"], E["outliers"].set_index("variavel")
        decisao(
            f"**manter os outliers.** São eventos reais (ônibus lotados, engavetamentos) e o maior `km` cabe na extensão da própria rodovia. "
            f"A regra do IQR removeria {mil(ef['n_removidas'])} acidentes ({pct_br(ef['pct_removidas'])}), levando junto "
            f"{pct_br(ef['pct_fatais_perdidos'])} dos casos graves; acima do limite em `pessoas`, a taxa de gravidade é "
            f"{pct_br(op.loc['pessoas', 'taxa_fatal_acima_pct'])} contra {pct_br(op.loc['pessoas', 'taxa_fatal_abaixo_pct'])} abaixo. "
            "Remover apagaria justamente o sinal que interessa. Como o impacto é tratado: modelos baseados em árvore, que não sofrem com escala.")

    with sub[3]:
        st.subheader("Correlação")
        metodo = st.radio("Método (variáveis numéricas)", ["Pearson", "Spearman"], horizontal=True)
        corr = E["corr_pearson"] if metodo == "Pearson" else E["corr_spearman"]
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=560, margin=dict(t=10, b=10))
        mostrar_plot(fig)
        st.markdown("**V de Cramér** (associação entre categóricas, 0 a 1; corrigido para viés):")
        fig = px.imshow(E["cramers"], text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(height=520, margin=dict(t=10, b=10))
        mostrar_plot(fig)
        v_alvo = E["cramers"]["alvo"].drop("alvo").sort_values(ascending=False)
        preditoras = E["cramers"].drop(index="alvo", columns="alvo")
        pares = (preditoras.where(np.triu(np.ones(preditoras.shape, dtype=bool), k=1)).stack().sort_values(ascending=False).head(3))
        txt_pares = "; ".join(f"{a} × {b} ({br(v, 2)})" for (a, b), v in pares.items())
        decisao(
            f"`feridos = feridos_leves + feridos_graves` é uma identidade exata (correlação {br(E['corr_pearson'].loc['feridos', 'feridos_leves'], 2)} e "
            f"{br(E['corr_pearson'].loc['feridos', 'feridos_graves'], 2)}): redundante, e as três são vazamento — saem. "
            f"As categóricas mais associadas ao alvo são `{v_alvo.index[0]}` (V = {br(v_alvo.iloc[0], 2)}) e `{v_alvo.index[1]}` (V = {br(v_alvo.iloc[1], 2)}); "
            f"`{v_alvo.index[-1]}` quase não se associa (V = {br(v_alvo.iloc[-1], 2)}). "
            f"Pares mais redundantes entre as preditoras: {txt_pares}; mantidos, pois os modelos de árvore lidam com redundância.")

    with sub[4]:
        st.subheader("Linearidade: taxa de gravidade por faixa de cada variável")
        c1, c2 = st.columns(2)
        c1.plotly_chart(linha_taxas(E["linearidade"]["hora"], "Por hora do dia"), key=f"plot_{next(_contador)}", **_W_PLOT)
        c2.plotly_chart(linha_taxas(E["linearidade"]["km"], "Por faixa de km (decis)"), key=f"plot_{next(_contador)}", **_W_PLOT)
        c3, c4 = st.columns(2)
        c3.plotly_chart(linha_taxas(E["linearidade"]["veiculos"], "Por nº de veículos"), key=f"plot_{next(_contador)}", **_W_PLOT)
        c4.plotly_chart(linha_taxas(E["linearidade"]["pessoas"], "Por nº de pessoas"), key=f"plot_{next(_contador)}", **_W_PLOT)
        res = E["linearidade_resumo"].set_index("variavel")
        mostrar_df(res.round(2))
        h = res.loc["hora"]
        mono = [v for v in ["veiculos", "pessoas", "km"] if abs(res.loc[v, "rho_monotonia"]) >= 0.9]
        decisao(
            f"a taxa por hora **não é monótona**: máxima às {h['faixa_maior_taxa']}h ({pct_br(h['taxa_max_pct'])}), mínima às {h['faixa_menor_taxa']}h "
            f"({pct_br(h['taxa_min_pct'])}), com patamar alto na madrugada e à noite. Já {', '.join('`' + v + '`' for v in mono) or 'nenhuma variável'} cresce de forma "
            "monótona, mas não linear no logit. Uma regressão logística só capturaria isso com engenharia manual; por isso entra como **baseline** e o "
            "modelo principal é **baseado em árvores (gradient boosting)**, que absorve não linearidades e interações (ex.: tipo de acidente × tipo de pista). "
            "Criamos `madrugada` (0h–5h) e `fim_de_semana` como apoio ao baseline.")

    with sub[5]:
        st.subheader("Cardinalidade das categóricas")
        mostrar_df(E["cardinalidade"].round(1), hide_index=True)
        mc = E["municipio_cauda"]
        fig = px.histogram(x=mc["contagens"], nbins=60, log_y=True,
                           labels={"x": "acidentes por município", "y": "municípios"}, title="Acidentes por município")
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        mostrar_plot(fig)
        decisao(
            f"`municipio` tem {mil(mc['n'])} categorias e {pct_br(mc['pct_municipios_raros'], 0)} delas têm menos de 30 acidentes (somam só "
            f"{pct_br(mc['pct_acidentes_em_raros'], 0)} dos registros). One-hot criaria ~{mil(mc['n'])} colunas quase vazias. "
            f"**Target encoding** (com validação cruzada interna, que evita vazamento) para `municipio`, `causa_acidente` ({limpo['causa_acidente'].nunique()}) e `br` ({limpo['br'].nunique()}). "
            "Alternativa descartada: one-hot e agrupar raros em «Outros» (perde a informação de municípios perigosos). "
            "Colunas com poucas categorias usam one-hot; `tracado_via` (605 combinações) já virou 12 colunas 0/1 na limpeza.")

    with sub[6]:
        st.subheader("Análise temporal")
        T = E["temporal"]
        mostrar_plot(barra_taxas(T["hora"]))
        c1, c2 = st.columns(2)
        c1.plotly_chart(barra_taxas(T["dia_semana"], 300), key=f"plot_{next(_contador)}", **_W_PLOT)
        c2.plotly_chart(barra_taxas(T["mes"], 300), key=f"plot_{next(_contador)}", **_W_PLOT)
        dia, mes = T["dia_semana"], T["mes"]
        decisao(
            f"a gravidade varia muito mais ao longo do dia (de {pct_br(T['hora']['taxa_pct'].min())} a {pct_br(T['hora']['taxa_pct'].max())}) "
            f"do que entre dias da semana ({pct_br(dia['taxa_pct'].min())} a {pct_br(dia['taxa_pct'].max())}) ou meses "
            f"({pct_br(mes['taxa_pct'].min())} a {pct_br(mes['taxa_pct'].max())}). Variáveis criadas: `hora`, `mes`, `fim_de_semana` (sábado/domingo) e `madrugada` "
            "(0h–5h). Não usamos o ano (só 2025).")

    with sub[7]:
        st.subheader("Análise geográfica")
        modo = st.radio("Mapa de calor", ["Acidentes graves", "Todos os acidentes"], horizontal=True)
        pts = limpo.loc[y.eq(1) if modo == "Acidentes graves" else slice(None), ["latitude", "longitude"]].dropna()
        if len(pts) > 25000:
            pts = pts.sample(25000, random_state=SEED)
        if hasattr(px, "density_map"):
            mapa = px.density_map(pts, lat="latitude", lon="longitude", radius=7, center=dict(lat=-14, lon=-52), zoom=3,
                                  map_style="carto-positron", height=520)
        else:
            mapa = px.density_mapbox(pts, lat="latitude", lon="longitude", radius=7, center=dict(lat=-14, lon=-52), zoom=3,
                                     mapbox_style="carto-positron", height=520)
        mapa.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        mostrar_plot(mapa)
        c1, c2 = st.columns(2)
        uf = E["geo_uf"]
        c1.markdown("**Taxa de gravidade por UF**")
        c1.plotly_chart(barra_taxas(uf, 360), key=f"plot_{next(_contador)}", **_W_PLOT)
        c2.markdown("**BRs com maior taxa (300+ acidentes)**")
        with c2:
            mostrar_df(E["geo_br"].head(15).round(1).rename(columns={"grupo": "BR"}), hide_index=True)
        decisao(
            "`uf`, `br`, `latitude` e `longitude` entram no modelo: a letalidade se concentra em regiões específicas (H5), não só onde há mais acidentes. "
            "Os modelos de árvore fazem cortes espaciais diretamente sobre latitude/longitude. Alternativa não adotada nesta versão: criar "
            "macro-região ou cluster de trecho (DBSCAN em BR + km), registrada como melhoria futura.")


# ===========================================================================
# ABA 3 — MODELAGEM
# ===========================================================================
with tabs[2]:
    st.header("Modelagem")
    ft = preparar_features(limpo.head(5))[1]
    st.markdown(
        f"**Alvo:** `{alvo_nome}` · **Divisão:** 80% treino / 20% teste, estratificada (`random_state=42`).\n\n"
        "**Pipeline (tudo dentro do treino, sem vazamento):** numéricas → mediana; categóricas com poucas categorias → one-hot "
        "(categorias raras agrupadas); `municipio`, `causa_acidente` e `br` → target encoding; nulos categóricos → «Desconhecido».\n\n"
        "**Modelos:** baseline = Regressão Logística · principal = HistGradientBoosting (e Random Forest como comparação) · "
        "referência = Dummy. Hiperparâmetros: `RandomizedSearchCV` com validação cruzada estratificada (3 dobras), "
        "otimizando PR-AUC.")
    with st.expander("Variáveis de entrada e variáveis excluídas"):
        st.markdown(f"**Numéricas ({len(ft)}):** {', '.join(f'`{c}`' for c in ft)}\n\n"
                    f"**Categóricas (one-hot):** {', '.join(f'`{c}`' for c in CAT_BAIXA)}\n\n"
                    f"**Categóricas (target encoding):** {', '.join(f'`{c}`' for c in CAT_ALTA)}\n\n"
                    f"**Excluídas por vazamento:** {', '.join(f'`{c}`' for c in COLUNAS_VAZAMENTO)}\n\n"
                    f"**Excluídas por outro motivo:** {', '.join(f'`{c}`' for c in COLUNAS_DESCARTADAS)}")

    c1, c2, c3 = st.columns(3)
    rapido = c1.checkbox("Modo rápido (recomendado)", value=True)
    usar_busca = c2.checkbox("Rodar busca de hiperparâmetros", value=True)
    n_iter = c3.number_input("Combinações testadas na busca", min_value=2, max_value=30, value=8, step=1)
    st.caption("**Modo rápido:** a busca de hiperparâmetros usa uma amostra estratificada de 40% do treino (até 6 combinações) e o melhor "
               "conjunto é reajustado no treino completo; Random Forest com 100 árvores; importância calculada em 3.000 acidentes. "
               "**Modo completo:** busca no treino inteiro, 200 árvores e importância em 5.000 acidentes. "
               "Nos dois modos, busca e validação cruzada usam todos os núcleos da CPU, e uma configuração já treinada não é treinada de novo.")
    cfg_atual = (alvo_nome, usar_busca, int(n_iter), rapido)
    if st.button("Treinar modelos", type="primary"):
        memoria = memoria_modelos()
        if (chave,) + cfg_atual not in memoria:
            barra = st.progress(0.0, text="Preparando…")
            memoria[(chave,) + cfg_atual] = treinar_tudo(
                limpo, alvo_nome, usar_busca=usar_busca, n_iter=int(n_iter), rapido=rapido,
                progresso=lambda f, t: barra.progress(min(float(f), 1.0), text=t))
            barra.empty()
        st.session_state["cfg_modelo"] = cfg_atual

    cfg = st.session_state.get("cfg_modelo")
    res = memoria_modelos().get((chave,) + tuple(cfg)) if cfg else None
    if not cfg:
        st.info("Clique em **Treinar modelos** para ver os resultados.")
    elif cfg[0] != alvo_nome:
        st.warning("O alvo mudou desde o último treino. Clique em **Treinar modelos** novamente.")
    elif res is None:
        st.info("Os resultados salvos foram descartados (cache limpo). Clique em **Treinar modelos** novamente.")
    else:
        st.success(f"Treino concluído em {br(res['tempo_s'], 0)} s — modo {res['modo']}, {res['nucleos']} núcleo(s) de CPU.")
        y_te = res["y_te"]
        prev = float(y_te.mean() * 100)
        sp = res["split"]
        st.caption(f"Treino: {mil(sp['n_treino'])} acidentes ({pct_br(sp['pct_fatal_treino'])} graves) · Teste: {mil(sp['n_teste'])} "
                   f"({pct_br(sp['pct_fatal_teste'])} graves)")

        st.subheader("Comparação dos modelos (limiar 0,50, conjunto de teste)")
        comp = res["comparacao"].rename(columns={"acuracia": "acurácia", "precisao": "precisão", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})
        try:
            mostrar_df(comp.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"))
        except Exception:
            mostrar_df(comp.round(3))
        st.caption("Precisão, recall e F1 referem-se à classe grave. A acurácia sozinha engana: o Dummy tem a maior acurácia e não detecta nenhum caso grave.")

        if res["busca_tab"] is not None:
            with st.expander("Busca de hiperparâmetros (RandomizedSearchCV, 3 dobras, PR-AUC)"):
                st.markdown("**Melhores parâmetros:** " + ", ".join(f"`{k}={v}`" for k, v in res["melhores_params"].items()))
                mostrar_df(res["busca_tab"].round(4), hide_index=True)

        st.subheader("Curva precisão × recall")
        fig = go.Figure()
        for nome, proba in res["probas"].items():
            if nome.startswith("Dummy"):
                continue
            r, p = curva_pr(y_te, proba)
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=nome))
        fig.add_hline(y=prev / 100, line_dash="dash", annotation_text="acaso (prevalência)")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precisão", height=420, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=-0.25))
        mostrar_plot(fig)

        st.subheader("Ponto de operação")
        nomes = list(res["probas"])
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Modelo", nomes, index=nomes.index(res["principal"]))
        modo = c2.radio("Limiar de decisão", ["0,50 (padrão)", f"Ótimo por validação cruzada no treino ({br(res['limiar_cv'], 2)})", "Manual"])
        if modo.startswith("0,50"):
            limiar = 0.5
        elif modo.startswith("Ótimo"):
            limiar = res["limiar_cv"]
        else:
            limiar = st.slider("Limiar", 0.05, 0.95, 0.5, 0.01)
        if sel != res["principal"] and modo.startswith("Ótimo"):
            st.caption("O limiar ótimo foi calculado para o modelo principal; para os demais é apenas referência.")
        proba = res["probas"][sel]
        m = metricas(y_te, proba, limiar)
        tn, fp, fn, tp = matriz_confusao(y_te, proba, limiar).ravel()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Precisão", pct_br(m["precisao"] * 100))
        k2.metric("Recall", pct_br(m["recall"] * 100))
        k3.metric("F1", br(m["f1"], 3))
        k4.metric("PR-AUC", br(m["pr_auc"], 3))
        fig = px.imshow(np.array([[tn, fp], [fn, tp]]), text_auto=True, color_continuous_scale="Blues",
                        x=["Previsto: não grave", "Previsto: grave"], y=["Real: não grave", "Real: grave"])
        fig.update_layout(height=340, margin=dict(t=10, b=10), coloraxis_showscale=False)
        mostrar_plot(fig)
        lift = m["precisao"] * 100 / prev if prev else float("nan")
        st.markdown(
            f"**Interpretação:** com limiar {br(limiar, 2)}, `{sel}` identifica **{pct_br(m['recall'] * 100, 0)}** dos acidentes graves "
            f"({mil(tp)} de {mil(tp + fn)}). Dos {mil(tp + fp)} acidentes que ele sinaliza, **{pct_br(m['precisao'] * 100, 0)}** são realmente graves: "
            f"{br(lift, 1)}× a taxa de {pct_br(prev)} de um sorteio ao acaso. Em compensação, {mil(fp)} acidentes não graves "
            f"({pct_br(fp / (tn + fp) * 100, 0)}) geram alarme falso. Na prática, o modelo serve para **priorizar combinações de risco** "
            "(fiscalização e sinalização), não para prever o desfecho de um acidente individual.")

        st.subheader("O que mais pesa no modelo principal")
        imp = res["importancias"].head(15).iloc[::-1]
        fig = px.bar(imp, x="queda_pr_auc", y="variavel", orientation="h", error_x="desvio",
                     labels={"queda_pr_auc": "queda da PR-AUC ao embaralhar a variável", "variavel": ""})
        fig.update_layout(height=470, margin=dict(t=10, b=10))
        mostrar_plot(fig)
        n_imp, rep_imp = ("3.000", 2) if res["modo"] == "rápido" else ("5.000", 3)
        st.caption(f"Importância por permutação no conjunto de teste (amostra de até {n_imp} acidentes), {rep_imp} repetições.")

    with st.expander("Limitações e melhorias futuras"):
        st.markdown(
            "- Só um ano (2025) e só rodovias federais: não generaliza para vias municipais ou estaduais.\n"
            "- `tipo_acidente` e `causa_acidente` só são conhecidos após o acidente; o modelo descreve risco associado, não prevê antes da ocorrência.\n"
            "- Associação não é causalidade (álcool, velocidade e visibilidade não são medidos diretamente).\n"
            f"- {pct_br(limpo['contagem_inconsistente'].mean() * 100)} dos registros têm contagem de pessoas inconsistente; testar o modelo sem essas linhas.\n"
            "- Melhorias: SHAP para explicar casos individuais, cluster de trechos (BR + km), vários anos de dados e calibração das probabilidades.")


# ===========================================================================
# ABA 4 — O QUE DEU ERRADO
# ===========================================================================
with tabs[3]:
    st.header("O que deu errado (e o que aprendemos)")
    st.info("**Demonstrações, não histórico do grupo.** As seções «Demonstração 1, 2 e 3» abaixo são demonstrações reproduzíveis de erros clássicos, "
            "rodadas aqui para mostrar o problema com números. A tentativa que **realmente aconteceu** com o grupo deve ser escrita na caixa abaixo: "
            "é ela que vai para o slide, e as demonstrações servem de apoio.")
    st.subheader("A tentativa que realmente aconteceu com o grupo")
    st.text_area("O que foi tentado, por que deu errado e o que o time aprendeu", key="tentativa_real", height=140)
    st.caption("O texto fica só nesta sessão do navegador: copie para o slide. Exemplos que de fato ocorreram durante o projeto "
               "(confirmem se foram vividos por vocês): erro de permissão ao criar a pasta `dados` no Windows; treino de ~2 minutos "
               "que levou ao modo rápido; textos «N/A» lidos como categorias em regional, delegacia e uop.")
    st.divider()
    st.subheader(f"Demonstração 1: «o modelo acertou {pct_br(100 - geral)}» só prevendo sempre «não grave»")
    acc = 100 - geral
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia do Dummy", pct_br(acc))
    c2.metric("Recall da classe grave", "0%")
    c3.metric("Acidentes graves detectados", "0")
    st.markdown(f"**Aprendizado:** com {pct_br(geral)} de casos graves, acertar {pct_br(acc)} não prova nada. Passamos a avaliar por "
                "precisão, recall, F1 e PR-AUC da classe grave, e a comparar sempre com o Dummy.")

    st.subheader("Demonstração 2: oversampling por duplicação de linhas antes de separar treino e teste")
    amostra = st.checkbox("Usar amostra de 30 mil acidentes (mais rápido)", value=True)
    if st.button("Rodar demonstração do oversampling"):
        with st.spinner("Treinando duas Random Forests…"):
            st.session_state["demo_over"] = (alvo_nome, amostra, rodar_oversampling(limpo, chave, alvo_nome, amostra))
    d = st.session_state.get("demo_over")
    if d and d[0] == alvo_nome:
        mostrar_df(d[2].rename(columns={"acuracia": "acurácia", "precisao": "precisão", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}).round(3))
        errado, certo = d[2].iloc[0], d[2].iloc[1]
        st.markdown(f"**Aprendizado:** duplicando os casos graves antes da divisão, cópias dos mesmos acidentes caem no treino e no teste: o F1 vai a "
                    f"{br(errado['f1'], 2)}, contra {br(certo['f1'], 2)} quando o balanceamento é feito só no treino e avaliado no teste real. "
                    "O resultado inflado seria «memorização», não aprendizado. Aqui o oversampling é por duplicação de linhas; o SMOTE, que cria vizinhos sintéticos em vez de cópias, tem o mesmo problema se for aplicado antes da divisão. "
                    "Regra do grupo: dividir primeiro, balancear depois, só no treino (ou usar `class_weight`).")

    st.subheader("Demonstração 3: colunas de vazamento (feridos, ilesos, ignorados) como variáveis de entrada")
    if st.button("Rodar demonstração do vazamento"):
        with st.spinner("Treinando…"):
            st.session_state["demo_vaza"] = (alvo_nome, rodar_vazamento(limpo, chave, alvo_nome))
    d = st.session_state.get("demo_vaza")
    if d and d[0] == alvo_nome:
        mostrar_df(d[1].rename(columns={"acuracia": "acurácia", "precisao": "precisão", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}).round(3))
        st.markdown(f"**Aprendizado:** o F1 salta de {br(d[1].iloc[0]['f1'], 2)} para {br(d[1].iloc[1]['f1'], 2)} só por incluir colunas que descrevem o desfecho. "
                    "Resultado bom demais é sinal de alerta: o modelo «lê a resposta» e seria inútil para identificar risco antes do acidente.")

        