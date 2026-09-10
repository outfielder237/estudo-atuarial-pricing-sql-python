# Bibliotecas

import numpy as np
import pandas as pd
from pathlib import Path
SEMENTE = 20260906
rng = np.random.default_rng(SEMENTE)

# Tamanho da carteira

N_APOLICES = 100_000
id_apolice = np.arange(1, N_APOLICES + 1)

# Exposição

exposicao = rng.uniform(
    low=1/12,
    high=1,
    size=N_APOLICES
)

exposicao = np.round(exposicao, 4)

# Idade do condutor

idade_condutor = np.clip(
    rng.normal(
        loc=42,
        scale=13,
        size=N_APOLICES
    ),
    18,
    80
).round().astype(int)

# Idade do veículo

idade_veiculo = np.clip(
    rng.gamma(
        shape=2.5,
        scale=3,
        size=N_APOLICES
    ),
    0,
    20
).round().astype(int)

# Região brasileira

regioes = [
    "Sudeste",
    "Sul",
    "Nordeste",
    "Centro-Oeste",
    "Norte"
]

prob_regiao = [
    0.48,
    0.18,
    0.18,
    0.10,
    0.06
]

regiao = rng.choice(
    regioes,
    size=N_APOLICES,
    p=prob_regiao
)

# Tipo de veículo

tipos_veiculo = [
    "Hatch",
    "Sedan",
    "SUV",
    "Picape",
    "Utilitario"
]

prob_tipo_veiculo = [
    0.30,
    0.25,
    0.25,
    0.12,
    0.08
]

tipo_veiculo = rng.choice(
    tipos_veiculo,
    size=N_APOLICES,
    p=prob_tipo_veiculo
)

# Tipo de uso

tipos_uso = [
    "Particular",
    "Comercial"
]

tipo_uso = rng.choice(
    tipos_uso,
    size=N_APOLICES,
    p=[0.88, 0.12]
)

# Cobertura

coberturas = [
    "Basica",
    "Intermediaria",
    "Completa"
]

nivel_cobertura = rng.choice(
    coberturas,
    size=N_APOLICES,
    p=[0.25, 0.45, 0.30]
)

# Classe de bônus

classe_bonus = rng.choice(
    np.arange(0, 11),
    size=N_APOLICES,
    p=[
        0.08,
        0.08,
        0.09,
        0.10,
        0.11,
        0.11,
        0.11,
        0.10,
        0.09,
        0.07,
        0.06
    ]
)

# Tabela de apólices_1

apolices = pd.DataFrame({
    "id_apolice": id_apolice,
    "exposicao": exposicao,
    "idade_condutor": idade_condutor,
    "idade_veiculo": idade_veiculo,
    "regiao": regiao,
    "tipo_veiculo": tipo_veiculo,
    "tipo_uso": tipo_uso,
    "nivel_cobertura": nivel_cobertura,
    "classe_bonus": classe_bonus
})

apolices.head()
apolices.info()

# Frequência atuarial

eta_freq = np.full(
    N_APOLICES,
    -3.0
)

# Efeito da idade do condutor

eta_freq += np.where(
    idade_condutor < 25,
    0.35,
    0
)

eta_freq += np.where(
    (idade_condutor >= 25) &
    (idade_condutor <= 65),
    0,
    0
)

# Efeito adicional para condutores acima de 65 anos
# No universo sintético do projeto, assumimos um acréscimo de 0,20
# no preditor linear da frequência para esse grupo.
eta_freq += np.where(
    idade_condutor > 65,
    0.20,
    0
)

# Efeito do uso comercial

eta_freq += np.where(
    tipo_uso == "Comercial",
    0.30,
    0
)

# Efeito da região

efeito_regiao_freq = {
    "Sudeste": 0.10,
    "Sul": 0.00,
    "Nordeste": -0.05,
    "Centro-Oeste": 0.08,
    "Norte": 0.03
}

eta_freq += pd.Series(regiao).map(
    efeito_regiao_freq
).to_numpy()

# Efeito da classe de bônus

eta_freq += -0.055 * classe_bonus

# Efeito da idade do veículo

eta_freq += 0.012 * idade_veiculo

# Frequência anual esperada

lambda_anual = np.exp(eta_freq)

apolices["lambda_verdadeiro"] = lambda_anual

apolices["lambda_verdadeiro"].describe()

# Incorporar a exposição

media_sinistros = (
    lambda_anual *
    exposicao
)

# Gerar número de sinistros

numero_sinistros = rng.poisson(
    media_sinistros
)

apolices["numero_sinistros"] = numero_sinistros

apolices[
    "numero_sinistros"
].value_counts().sort_index()

# Frequência observada da carteira

frequencia_observada = (
    apolices["numero_sinistros"].sum()
    /
    apolices["exposicao"].sum()
)

print(
    "Frequência anual observada:",
    round(frequencia_observada, 4)
)

# Severidade

eta_sev = np.full(
    N_APOLICES,
    np.log(7_500)
)

# Tipo de veículo

efeito_veiculo_sev = {
    "Hatch": 0.00,
    "Sedan": 0.08,
    "SUV": 0.22,
    "Picape": 0.30,
    "Utilitario": 0.18
}

eta_sev += pd.Series(
    tipo_veiculo
).map(
    efeito_veiculo_sev
).to_numpy()

# Efeito da cobertura

efeito_cobertura_sev = {
    "Basica": -0.10,
    "Intermediaria": 0.00,
    "Completa": 0.18
}

eta_sev += pd.Series(
    nivel_cobertura
).map(
    efeito_cobertura_sev
).to_numpy()

# Efeito da idade do veículo

eta_sev += -0.008 * idade_veiculo

# Severidade esperada

mu_severidade = np.exp(
    eta_sev
)

apolices["severidade_verdadeira"] = (
    mu_severidade
)

apolices[
    "severidade_verdadeira"
].describe()

# Tabela de sinistros

indices_com_sinistro = np.repeat(
    np.arange(N_APOLICES),
    numero_sinistros
)

N_SINISTROS = len(
    indices_com_sinistro
)

print(
    "Total de sinistros:",
    N_SINISTROS
)

# Severidades individuais

shape_gamma = 2.2

mu_claim = mu_severidade[
    indices_com_sinistro
]

scale_claim = (
    mu_claim /
    shape_gamma
)

valor_sinistro = rng.gamma(
    shape=shape_gamma,
    scale=scale_claim
)

valor_sinistro = np.round(
    valor_sinistro,
    2
)

# Tabela de sinistros

sinistros = pd.DataFrame({
    "id_sinistro": np.arange(
        1,
        N_SINISTROS + 1
    ),
    "id_apolice": apolices.loc[
        indices_com_sinistro,
        "id_apolice"
    ].to_numpy(),
    "valor_sinistro": valor_sinistro
})

sinistros.head()

# Prêmio vigente

premio_puro_verdadeiro = (
    lambda_anual *
    mu_severidade
)

premio_base = (
    premio_puro_verdadeiro
    / (1 - 0.25)
)

erro_tarifa = rng.lognormal(
    mean=0,
    sigma=0.15,
    size=N_APOLICES
)

premio_vigente = (
    premio_base *
    erro_tarifa
)

apolices["premio_vigente"] = np.round(
    premio_vigente,
    2
)

# Versão operacional 

colunas_publicas = [
    "id_apolice",
    "exposicao",
    "idade_condutor",
    "idade_veiculo",
    "regiao",
    "tipo_veiculo",
    "tipo_uso",
    "nivel_cobertura",
    "classe_bonus",
    "premio_vigente"
]

apolices_modelagem = (
    apolices[
        colunas_publicas
    ].copy()
)

apolices_modelagem["premio_ganho"] = (
    apolices_modelagem["premio_vigente"]
    *
    apolices_modelagem["exposicao"]
)

# Salvar as bases

PASTA_DADOS = Path(
    "../data"
)

PASTA_DADOS.mkdir(
    parents=True,
    exist_ok=True
)

apolices_modelagem.to_csv(
    PASTA_DADOS / "apolices.csv",
    index=False,
    encoding="utf-8-sig"
)

sinistros.to_csv(
    PASTA_DADOS / "sinistros.csv",
    index=False,
    encoding="utf-8-sig"
)

# Testes de integridade

assert (
    apolices_modelagem[
        "id_apolice"
    ].is_unique
)

assert (
    apolices_modelagem[
        "exposicao"
    ] > 0
).all()

assert (
    sinistros[
        "valor_sinistro"
    ] > 0
).all()

assert set(
    sinistros["id_apolice"]
).issubset(
    set(
        apolices_modelagem[
            "id_apolice"
        ]
    )
)

# Resumo atuarial da carteira

exposicao_total = (
    apolices_modelagem[
        "exposicao"
    ].sum()
)

numero_total_sinistros = len(
    sinistros
)

custo_total_sinistros = (
    sinistros[
        "valor_sinistro"
    ].sum()
)

frequencia = (
    numero_total_sinistros
    /
    exposicao_total
)

severidade = (
    custo_total_sinistros
    /
    numero_total_sinistros
)

premio_total = (
    apolices_modelagem[
        "premio_ganho"
    ].sum()
)

loss_ratio = (
    custo_total_sinistros
    /
    premio_total
)

print(
    f"Apólices: {len(apolices_modelagem):,}"
)

print(
    f"Exposição: {exposicao_total:,.1f}"
)

print(
    f"Sinistros: {numero_total_sinistros:,}"
)

print(
    f"Frequência: {frequencia:.2%}"
)

print(
    f"Severidade média: R$ {severidade:,.2f}"
)

print(
    f"Custo de sinistros: R$ {custo_total_sinistros:,.2f}"
)

print(
    f"Prêmio ganho aproximado: R$ {premio_total:,.2f}"
)

print(
    f"Loss Ratio: {loss_ratio:.2%}"
)

print("\nDistribuição do número de sinistros por apólice:")

print(
    apolices["numero_sinistros"]
    .value_counts()
    .sort_index()
)

print("\nDistribuição da severidade:")

print(
    sinistros["valor_sinistro"]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

sinistros_por_apolice = (
    sinistros
    .groupby("id_apolice")
    .agg(
        quantidade_sinistros=(
            "id_sinistro",
            "count"
        ),
        sinistro_total=(
            "valor_sinistro",
            "sum"
        )
    )
    .reset_index()
)

base_check = (
    apolices_modelagem
    .merge(
        sinistros_por_apolice,
        on="id_apolice",
        how="left"
    )
)

base_check[
    "quantidade_sinistros"
] = (
    base_check[
        "quantidade_sinistros"
    ].fillna(0)
)

base_check[
    "sinistro_total"
] = (
    base_check[
        "sinistro_total"
    ].fillna(0)
)

resumo_regiao = (
    base_check
    .groupby("regiao")
    .agg(
        apolices=("id_apolice", "count"),
        exposicao=("exposicao", "sum"),
        sinistros=(
            "quantidade_sinistros",
            "sum"
        ),
        premio_ganho=(
            "premio_ganho",
            "sum"
        ),
        custo_sinistros=(
            "sinistro_total",
            "sum"
        )
    )
)

resumo_regiao[
    "frequencia"
] = (
    resumo_regiao["sinistros"]
    /
    resumo_regiao["exposicao"]
)

resumo_regiao[
    "severidade"
] = (
    resumo_regiao["custo_sinistros"]
    /
    resumo_regiao["sinistros"]
)

resumo_regiao[
    "loss_ratio"
] = (
    resumo_regiao["custo_sinistros"]
    /
    resumo_regiao["premio_ganho"]
)

print("\nResumo atuarial por região:")
print(resumo_regiao)

resumo_uso = (
    base_check
    .groupby("tipo_uso")
    .agg(
        apolices=("id_apolice", "count"),
        exposicao=("exposicao", "sum"),
        sinistros=("quantidade_sinistros", "sum"),
        premio_ganho=("premio_ganho", "sum"),
        custo_sinistros=("sinistro_total", "sum")
    )
)

resumo_uso["frequencia"] = (
    resumo_uso["sinistros"]
    /
    resumo_uso["exposicao"]
)

resumo_uso["severidade"] = (
    resumo_uso["custo_sinistros"]
    /
    resumo_uso["sinistros"]
)

resumo_uso["loss_ratio"] = (
    resumo_uso["custo_sinistros"]
    /
    resumo_uso["premio_ganho"]
)

print("\nResumo atuarial por tipo de uso:")
print(resumo_uso)

resumo_bonus = (
    base_check
    .groupby("classe_bonus")
    .agg(
        apolices=("id_apolice", "count"),
        exposicao=("exposicao", "sum"),
        sinistros=("quantidade_sinistros", "sum"),
        premio_ganho=("premio_ganho", "sum"),
        custo_sinistros=("sinistro_total", "sum")
    )
)

resumo_bonus["frequencia"] = (
    resumo_bonus["sinistros"]
    /
    resumo_bonus["exposicao"]
)

resumo_bonus["severidade"] = (
    resumo_bonus["custo_sinistros"]
    /
    resumo_bonus["sinistros"]
)

resumo_bonus["loss_ratio"] = (
    resumo_bonus["custo_sinistros"]
    /
    resumo_bonus["premio_ganho"]
)

print("\nResumo atuarial por classe de bônus:")
print(resumo_bonus)

resumo_veiculo = (
    base_check
    .groupby("tipo_veiculo")
    .agg(
        apolices=("id_apolice", "count"),
        exposicao=("exposicao", "sum"),
        sinistros=("quantidade_sinistros", "sum"),
        premio_ganho=("premio_ganho", "sum"),
        custo_sinistros=("sinistro_total", "sum")
    )
)

resumo_veiculo["frequencia"] = (
    resumo_veiculo["sinistros"]
    /
    resumo_veiculo["exposicao"]
)

resumo_veiculo["severidade"] = (
    resumo_veiculo["custo_sinistros"]
    /
    resumo_veiculo["sinistros"]
)

resumo_veiculo["loss_ratio"] = (
    resumo_veiculo["custo_sinistros"]
    /
    resumo_veiculo["premio_ganho"]
)

print("\nResumo atuarial por tipo de veículo:")
print(resumo_veiculo)