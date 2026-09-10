import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# ETAPA 9 — ESTRATÉGIAS DE PRICING
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Comparar três estratégias de preço:
#
# 1. Growth
# 2. Margin
# 3. Capital-efficient
#
# A lógica é combinar:
#
# risco técnico
# + prêmio comercial
# + retenção esperada
# + resultado econômico esperado
# + proxy de consumo de capital
#
# IMPORTANTE:
# Esta etapa ainda NÃO calcula capital econômico via simulação
# da distribuição agregada de perdas. Isso virá nas etapas
# seguintes. Aqui usamos uma proxy de capital baseada em uma
# carga explícita sobre o custo esperado e/ou sobre a volatilidade
# relativa do risco.
#
# Todos os parâmetros comportamentais e estratégicos são
# didáticos e sintéticos.
# ============================================================


# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

PASTA_DADOS = Path("../data")
PASTA_TABELAS = Path("../outputs/tabelas")
PASTA_FIGURAS = Path("../outputs/figuras")
ARQUIVO_BANCO = PASTA_DADOS / "pricing.db"

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Parâmetros comerciais herdados
# ------------------------------------------------------------

DESPESA_FIXA_ANUAL = 40.00
COMISSAO_PCT = 0.12
TRIBUTOS_ENCARGOS_PCT = 0.0765
DESPESA_VARIAVEL_PCT = 0.08
MARGEM_ALVO_PCT = 0.05

SOMA_CARREGAMENTOS_PCT = (
    COMISSAO_PCT
    + TRIBUTOS_ENCARGOS_PCT
    + DESPESA_VARIAVEL_PCT
    + MARGEM_ALVO_PCT
)

DENOMINADOR_COMERCIAL = 1 - SOMA_CARREGAMENTOS_PCT

if DENOMINADOR_COMERCIAL <= 0:
    raise ValueError("Carregamentos percentuais inválidos.")


# ------------------------------------------------------------
# 3. Parâmetros do modelo comportamental sintético
# ------------------------------------------------------------

BETA0_RETENCAO = np.log(0.82 / (1 - 0.82))
BETA_PRECO = -4.00
BETA_BONUS = 0.04
BETA_COMERCIAL = -0.10
BETA_COMPLETA = 0.08

# Intercepto equivalente quando classe_bonus entra sem centralização
INTERCEPTO_EQUIVALENTE = (
    BETA0_RETENCAO
    - 5 * BETA_BONUS
)


# ------------------------------------------------------------
# 4. Parâmetros estratégicos
# ------------------------------------------------------------
#
# Growth:
#   preço mais agressivo para preservar volume.
#
# Margin:
#   preço acima do indicado técnico/comercial para capturar margem.
#
# Capital-efficient:
#   preço ajustado conforme proxy de intensidade de capital.
#
# O objetivo aqui NÃO é dizer que estes multiplicadores são ótimos.
# Eles são apenas cenários estratégicos comparáveis.

FATOR_GROWTH = 0.95
FATOR_MARGIN = 1.10

# Parâmetros da proxy de capital
CARGA_CAPITAL_BASE = 0.20
PESO_SEVERIDADE_CAPITAL = 0.50
PESO_FREQUENCIA_CAPITAL = 0.50

# Limites do fator capital-efficient para evitar preços extremos
FATOR_CAPITAL_MIN = 0.90
FATOR_CAPITAL_MAX = 1.20


# ------------------------------------------------------------
# 5. Leitura das bases
# ------------------------------------------------------------

conexao = sqlite3.connect(ARQUIVO_BANCO)

apolices = pd.read_sql_query(
    """
    SELECT
        id_apolice,
        exposicao,
        idade_condutor,
        idade_veiculo,
        regiao,
        tipo_veiculo,
        tipo_uso,
        nivel_cobertura,
        classe_bonus,
        premio_vigente,
        premio_ganho
    FROM apolices
    ORDER BY id_apolice;
    """,
    conexao
)

sinistros = pd.read_sql_query(
    """
    SELECT
        s.id_sinistro,
        s.id_apolice,
        s.valor_sinistro,
        a.idade_veiculo,
        a.tipo_veiculo,
        a.nivel_cobertura
    FROM sinistros s
    INNER JOIN apolices a
        ON s.id_apolice = a.id_apolice
    ORDER BY s.id_sinistro;
    """,
    conexao
)

conexao.close()

print("\n================ BASES LIDAS ================")
print(f"Apólices: {apolices.shape}")
print(f"Sinistros: {sinistros.shape}")


# ------------------------------------------------------------
# 6. Integridade
# ------------------------------------------------------------

if apolices.empty:
    raise ValueError("A base de apólices está vazia.")

if sinistros.empty:
    raise ValueError("A base de sinistros está vazia.")

if (apolices["exposicao"] <= 0).any():
    raise ValueError("Existem exposições <= 0.")

if (apolices["premio_vigente"] <= 0).any():
    raise ValueError("Existem prêmios vigentes <= 0.")


# ------------------------------------------------------------
# 7. GLM de frequência
# ------------------------------------------------------------

apolices["log_exposicao"] = np.log(apolices["exposicao"])

apolices["faixa_idade_modelo"] = pd.cut(
    apolices["idade_condutor"],
    bins=[17, 24, 65, 80],
    labels=["18-24", "25-65", "66+"],
    include_lowest=True
)

apolices["faixa_idade_modelo"] = pd.Categorical(
    apolices["faixa_idade_modelo"],
    categories=["25-65", "18-24", "66+"]
)

apolices["regiao"] = pd.Categorical(
    apolices["regiao"],
    categories=["Sul", "Sudeste", "Nordeste", "Centro-Oeste", "Norte"]
)

apolices["tipo_uso"] = pd.Categorical(
    apolices["tipo_uso"],
    categories=["Particular", "Comercial"]
)

sinistros_por_apolice = (
    sinistros
    .groupby("id_apolice")
    .agg(
        quantidade_sinistros=("id_sinistro", "count"),
        custo_total_sinistros=("valor_sinistro", "sum")
    )
    .reset_index()
)

base_freq = apolices.merge(
    sinistros_por_apolice,
    on="id_apolice",
    how="left"
)

base_freq["quantidade_sinistros"] = (
    base_freq["quantidade_sinistros"]
    .fillna(0)
    .astype(int)
)

base_freq["custo_total_sinistros"] = (
    base_freq["custo_total_sinistros"]
    .fillna(0.0)
)

formula_freq = """
quantidade_sinistros
~
C(faixa_idade_modelo, Treatment(reference="25-65"))
+
idade_veiculo
+
C(regiao, Treatment(reference="Sul"))
+
C(tipo_uso, Treatment(reference="Particular"))
+
classe_bonus
"""

modelo_freq = smf.glm(
    formula=formula_freq,
    data=base_freq,
    family=sm.families.Poisson(),
    offset=base_freq["log_exposicao"]
).fit()


# ------------------------------------------------------------
# 8. GLM de severidade
# ------------------------------------------------------------

sinistros["tipo_veiculo"] = pd.Categorical(
    sinistros["tipo_veiculo"],
    categories=["Hatch", "Sedan", "SUV", "Picape", "Utilitario"]
)

sinistros["nivel_cobertura"] = pd.Categorical(
    sinistros["nivel_cobertura"],
    categories=["Intermediaria", "Basica", "Completa"]
)

formula_sev = """
valor_sinistro
~
C(tipo_veiculo, Treatment(reference="Hatch"))
+
C(nivel_cobertura, Treatment(reference="Intermediaria"))
+
idade_veiculo
"""

modelo_sev = smf.glm(
    formula=formula_sev,
    data=sinistros,
    family=sm.families.Gamma(
        link=sm.families.links.Log()
    )
).fit()


# ------------------------------------------------------------
# 9. Base técnica
# ------------------------------------------------------------

base = base_freq.copy()

base["tipo_veiculo"] = pd.Categorical(
    base["tipo_veiculo"],
    categories=["Hatch", "Sedan", "SUV", "Picape", "Utilitario"]
)

base["nivel_cobertura"] = pd.Categorical(
    base["nivel_cobertura"],
    categories=["Intermediaria", "Basica", "Completa"]
)

base["sinistros_previstos_periodo"] = modelo_freq.predict(
    base,
    offset=base["log_exposicao"]
)

base["frequencia_prevista"] = (
    base["sinistros_previstos_periodo"]
    /
    base["exposicao"]
)

base["severidade_prevista"] = modelo_sev.predict(base)

base["premio_puro_anual"] = (
    base["frequencia_prevista"]
    *
    base["severidade_prevista"]
)

base["custo_esperado_periodo"] = (
    base["premio_puro_anual"]
    *
    base["exposicao"]
)

base["premio_comercial_indicado"] = (
    base["premio_puro_anual"] + DESPESA_FIXA_ANUAL
) / DENOMINADOR_COMERCIAL


# ------------------------------------------------------------
# 10. Proxy de intensidade de capital
# ------------------------------------------------------------
#
# A proxy combina frequência e severidade normalizadas.
#
# Quanto maior o risco relativo, maior a carga de capital.
#
# Isto ainda NÃO é capital econômico modelado.
# É apenas uma medida didática de intensidade de risco.

freq_media = base["frequencia_prevista"].mean()
sev_media = base["severidade_prevista"].mean()

base["indice_freq_relativa"] = (
    base["frequencia_prevista"]
    /
    freq_media
)

base["indice_sev_relativa"] = (
    base["severidade_prevista"]
    /
    sev_media
)

base["indice_capital_relativo"] = (
    PESO_FREQUENCIA_CAPITAL * base["indice_freq_relativa"]
    +
    PESO_SEVERIDADE_CAPITAL * base["indice_sev_relativa"]
)

base["capital_proxy_anual"] = (
    CARGA_CAPITAL_BASE
    *
    base["premio_puro_anual"]
    *
    base["indice_capital_relativo"]
)


# ------------------------------------------------------------
# 11. Construção do fator Capital-efficient
# ------------------------------------------------------------
#
# Riscos com intensidade de capital acima da média recebem
# maior fator de preço.
#
# Riscos abaixo da média recebem fator menor.

fator_capital = (
    1
    +
    0.20 * (
        base["indice_capital_relativo"]
        - 1
    )
)

base["fator_capital_efficient"] = np.clip(
    fator_capital,
    FATOR_CAPITAL_MIN,
    FATOR_CAPITAL_MAX
)


# ------------------------------------------------------------
# 12. Estratégias de preço
# ------------------------------------------------------------
#
# Growth:
# 95% do prêmio comercial indicado
#
# Margin:
# 110% do prêmio comercial indicado
#
# Capital-efficient:
# prêmio comercial indicado ajustado pela intensidade de capital

base["preco_growth"] = (
    base["premio_comercial_indicado"]
    *
    FATOR_GROWTH
)

base["preco_margin"] = (
    base["premio_comercial_indicado"]
    *
    FATOR_MARGIN
)

base["preco_capital_efficient"] = (
    base["premio_comercial_indicado"]
    *
    base["fator_capital_efficient"]
)


# ------------------------------------------------------------
# 13. Modelo comportamental para cada estratégia
# ------------------------------------------------------------
#
# Usamos o modelo estrutural sintético da Etapa 8:
#
# logit(p) =
# beta0_equivalente
# + beta_preco * log(preco / vigente)
# + beta_bonus * bonus
# + beta_comercial * I(Comercial)
# + beta_completa * I(Completa)

def logistic(x):
    return 1 / (1 + np.exp(-x))

indicador_comercial = (
    base["tipo_uso"].astype(str) == "Comercial"
).astype(int)

indicador_completa = (
    base["nivel_cobertura"].astype(str) == "Completa"
).astype(int)

def prob_retencao_para_preco(df, coluna_preco):
    log_rel_preco = np.log(
        df[coluna_preco]
        /
        df["premio_vigente"]
    )

    eta = (
        INTERCEPTO_EQUIVALENTE
        + BETA_PRECO * log_rel_preco
        + BETA_BONUS * df["classe_bonus"]
        + BETA_COMERCIAL * indicador_comercial
        + BETA_COMPLETA * indicador_completa
    )

    return logistic(eta)

base["retencao_growth"] = prob_retencao_para_preco(
    base,
    "preco_growth"
)

base["retencao_margin"] = prob_retencao_para_preco(
    base,
    "preco_margin"
)

base["retencao_capital_efficient"] = prob_retencao_para_preco(
    base,
    "preco_capital_efficient"
)


# ------------------------------------------------------------
# 14. Função econômica por estratégia
# ------------------------------------------------------------
#
# Para cada apólice:
#
# Receita esperada =
# preço * probabilidade de retenção
#
# Custo técnico esperado =
# prêmio puro * probabilidade de retenção
#
# Despesa fixa esperada =
# despesa fixa * probabilidade de retenção
#
# Custos percentuais =
# preço * retenção * (comissão + tributos + despesa variável)
#
# Resultado esperado =
# Receita - custos
#
# A margem-alvo NÃO é tratada como custo aqui.
# Ela é uma meta de resultado, não uma saída operacional real.
#
# RoC proxy =
# resultado esperado / capital esperado

CUSTOS_PERCENTUAIS_OPERACIONAIS = (
    COMISSAO_PCT
    + TRIBUTOS_ENCARGOS_PCT
    + DESPESA_VARIAVEL_PCT
)

def calcular_metricas_estrategia(
    df,
    nome,
    coluna_preco,
    coluna_retencao
):
    p = df[coluna_retencao]
    preco = df[coluna_preco]

    df[f"receita_esperada_{nome}"] = (
        preco * p
    )

    df[f"custo_risco_esperado_{nome}"] = (
        df["premio_puro_anual"] * p
    )

    df[f"despesa_fixa_esperada_{nome}"] = (
        DESPESA_FIXA_ANUAL * p
    )

    df[f"custos_percentuais_esperados_{nome}"] = (
        preco
        * p
        * CUSTOS_PERCENTUAIS_OPERACIONAIS
    )

    df[f"resultado_esperado_{nome}"] = (
        df[f"receita_esperada_{nome}"]
        - df[f"custo_risco_esperado_{nome}"]
        - df[f"despesa_fixa_esperada_{nome}"]
        - df[f"custos_percentuais_esperados_{nome}"]
    )

    df[f"capital_esperado_{nome}"] = (
        df["capital_proxy_anual"]
        * p
    )

    df[f"roc_proxy_{nome}"] = (
        df[f"resultado_esperado_{nome}"]
        /
        df[f"capital_esperado_{nome}"].replace(0, np.nan)
    )

    df[f"lr_tecnico_{nome}"] = (
        df["premio_puro_anual"]
        /
        preco
    )

    return df

for nome, preco, ret in [
    ("growth", "preco_growth", "retencao_growth"),
    ("margin", "preco_margin", "retencao_margin"),
    (
        "capital_efficient",
        "preco_capital_efficient",
        "retencao_capital_efficient"
    )
]:
    base = calcular_metricas_estrategia(
        base,
        nome,
        preco,
        ret
    )


# Checagem preventiva: confirma que todas as colunas calculadas
# existem antes de montar o resumo agregado.
for chave in ["growth", "margin", "capital_efficient"]:
    colunas_esperadas = [
        f"receita_esperada_{chave}",
        f"custo_risco_esperado_{chave}",
        f"despesa_fixa_esperada_{chave}",
        f"custos_percentuais_esperados_{chave}",
        f"resultado_esperado_{chave}",
        f"capital_esperado_{chave}",
        f"roc_proxy_{chave}",
        f"lr_tecnico_{chave}"
    ]

    faltantes = [
        coluna
        for coluna in colunas_esperadas
        if coluna not in base.columns
    ]

    if faltantes:
        raise KeyError(
            f"Colunas ausentes para a estratégia '{chave}': {faltantes}"
        )


# ------------------------------------------------------------
# 15. Resumo agregado por estratégia
# ------------------------------------------------------------

def resumo_estrategia(
    df,
    chave,
    rotulo,
    coluna_preco,
    coluna_retencao
):
    """
    chave:
        Identificador interno usado nos nomes das colunas:
        growth, margin ou capital_efficient.

    rotulo:
        Nome amigável exibido nos relatórios:
        Growth, Margin ou Capital-efficient.
    """

    apolices_esperadas = df[coluna_retencao].sum()

    receita_total = df[f"receita_esperada_{chave}"].sum()
    custo_risco_total = df[f"custo_risco_esperado_{chave}"].sum()
    despesa_fixa_total = df[f"despesa_fixa_esperada_{chave}"].sum()
    custos_pct_total = df[
        f"custos_percentuais_esperados_{chave}"
    ].sum()
    resultado_total = df[f"resultado_esperado_{chave}"].sum()
    capital_total = df[f"capital_esperado_{chave}"].sum()

    roc = (
        resultado_total / capital_total
        if capital_total > 0
        else np.nan
    )

    premio_medio = (
        (df[coluna_preco] * df[coluna_retencao]).sum()
        /
        apolices_esperadas
    )

    retencao_media = df[coluna_retencao].mean()

    lr_tecnico_agregado = (
        custo_risco_total
        /
        receita_total
    )

    resultado_por_apolice_retida = (
        resultado_total
        /
        apolices_esperadas
    )

    capital_por_apolice_retida = (
        capital_total
        /
        apolices_esperadas
    )

    return {
        "estrategia": rotulo,
        "chave_interna": chave,
        "retencao_media": retencao_media,
        "apolices_esperadas_retidas": apolices_esperadas,
        "premio_medio_retido": premio_medio,
        "receita_esperada_total": receita_total,
        "custo_risco_esperado_total": custo_risco_total,
        "despesa_fixa_esperada_total": despesa_fixa_total,
        "custos_percentuais_esperados_total": custos_pct_total,
        "resultado_esperado_total": resultado_total,
        "capital_proxy_total": capital_total,
        "roc_proxy": roc,
        "lr_tecnico_agregado": lr_tecnico_agregado,
        "resultado_por_apolice_retida": resultado_por_apolice_retida,
        "capital_por_apolice_retida": capital_por_apolice_retida
    }

resumo_estrategias = pd.DataFrame([
    resumo_estrategia(
        base,
        "growth",
        "Growth",
        "preco_growth",
        "retencao_growth"
    ),
    resumo_estrategia(
        base,
        "margin",
        "Margin",
        "preco_margin",
        "retencao_margin"
    ),
    resumo_estrategia(
        base,
        "capital_efficient",
        "Capital-efficient",
        "preco_capital_efficient",
        "retencao_capital_efficient"
    )
])

print("\n================ COMPARAÇÃO DAS ESTRATÉGIAS ================")
print(resumo_estrategias)


# ------------------------------------------------------------
# 16. Ranking das estratégias
# ------------------------------------------------------------

resumo_estrategias["rank_retencao"] = (
    resumo_estrategias["retencao_media"]
    .rank(ascending=False, method="min")
)

resumo_estrategias["rank_resultado"] = (
    resumo_estrategias["resultado_esperado_total"]
    .rank(ascending=False, method="min")
)

resumo_estrategias["rank_roc"] = (
    resumo_estrategias["roc_proxy"]
    .rank(ascending=False, method="min")
)

resumo_estrategias["rank_capital"] = (
    resumo_estrategias["capital_proxy_total"]
    .rank(ascending=True, method="min")
)

print("\n================ RANKING DAS ESTRATÉGIAS ================")
print(
    resumo_estrategias[
        [
            "estrategia",
            "rank_retencao",
            "rank_resultado",
            "rank_roc",
            "rank_capital"
        ]
    ]
)


# ------------------------------------------------------------
# 17. Resumos por segmento
# ------------------------------------------------------------

base["faixa_etaria_analise"] = pd.cut(
    base["idade_condutor"],
    bins=[17, 24, 34, 44, 54, 64, 80],
    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    include_lowest=True
)

def resumo_segmento_estrategia(
    df,
    coluna_segmento,
    nome,
    coluna_preco,
    coluna_retencao
):
    temp = (
        df.groupby(coluna_segmento, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            retencao_media=(coluna_retencao, "mean"),
            preco_medio=(coluna_preco, "mean"),
            premio_puro_medio=("premio_puro_anual", "mean"),
            receita_esperada=(
                f"receita_esperada_{nome}",
                "sum"
            ),
            resultado_esperado=(
                f"resultado_esperado_{nome}",
                "sum"
            ),
            capital_proxy=(
                f"capital_esperado_{nome}",
                "sum"
            )
        )
    )

    temp["roc_proxy"] = (
        temp["resultado_esperado"]
        /
        temp["capital_proxy"].replace(0, np.nan)
    )

    temp["lr_tecnico"] = (
        temp["premio_puro_medio"]
        /
        temp["preco_medio"]
    )

    temp.insert(0, "estrategia", nome)

    return temp

lista_resumos_segmentados = []

for segmento in [
    "regiao",
    "tipo_uso",
    "tipo_veiculo",
    "nivel_cobertura",
    "classe_bonus",
    "faixa_etaria_analise"
]:
    tabelas = []

    for nome, preco, ret in [
        ("growth", "preco_growth", "retencao_growth"),
        ("margin", "preco_margin", "retencao_margin"),
        (
            "capital_efficient",
            "preco_capital_efficient",
            "retencao_capital_efficient"
        )
    ]:
        tabela = resumo_segmento_estrategia(
            base,
            segmento,
            nome,
            preco,
            ret
        ).reset_index()

        tabela.insert(0, "segmento", segmento)
        tabela = tabela.rename(
            columns={segmento: "categoria"}
        )

        tabelas.append(tabela)

    consolidado_segmento = pd.concat(
        tabelas,
        ignore_index=True
    )

    consolidado_segmento.to_csv(
        PASTA_TABELAS / f"estrategias_por_{segmento}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    lista_resumos_segmentados.append(consolidado_segmento)

resumo_segmentado_total = pd.concat(
    lista_resumos_segmentados,
    ignore_index=True
)

resumo_segmentado_total.to_csv(
    PASTA_TABELAS / "estrategias_segmentadas_consolidado.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 18. Exportação da comparação final
# ------------------------------------------------------------

resumo_estrategias.to_csv(
    PASTA_TABELAS / "comparacao_estrategias_pricing.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 19. Exportação da base detalhada
# ------------------------------------------------------------

colunas_exportacao = [
    "id_apolice",
    "exposicao",
    "idade_condutor",
    "idade_veiculo",
    "regiao",
    "tipo_veiculo",
    "tipo_uso",
    "nivel_cobertura",
    "classe_bonus",
    "premio_vigente",
    "premio_puro_anual",
    "premio_comercial_indicado",
    "frequencia_prevista",
    "severidade_prevista",
    "indice_capital_relativo",
    "capital_proxy_anual",
    "fator_capital_efficient",
    "preco_growth",
    "preco_margin",
    "preco_capital_efficient",
    "retencao_growth",
    "retencao_margin",
    "retencao_capital_efficient",
    "resultado_esperado_growth",
    "resultado_esperado_margin",
    "resultado_esperado_capital_efficient",
    "capital_esperado_growth",
    "capital_esperado_margin",
    "capital_esperado_capital_efficient",
    "roc_proxy_growth",
    "roc_proxy_margin",
    "roc_proxy_capital_efficient"
]

base[colunas_exportacao].to_csv(
    PASTA_TABELAS / "base_estrategias_pricing.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 20. Gráfico — retenção por estratégia
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    resumo_estrategias["estrategia"],
    resumo_estrategias["retencao_media"]
)

plt.ylabel("Retenção média prevista")
plt.title("Retenção esperada por estratégia")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "retencao_por_estrategia.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 21. Gráfico — resultado esperado por estratégia
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    resumo_estrategias["estrategia"],
    resumo_estrategias["resultado_esperado_total"]
)

plt.ylabel("Resultado esperado total (R$)")
plt.title("Resultado econômico esperado por estratégia")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "resultado_por_estrategia.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 22. Gráfico — capital por estratégia
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    resumo_estrategias["estrategia"],
    resumo_estrategias["capital_proxy_total"]
)

plt.ylabel("Capital proxy esperado (R$)")
plt.title("Consumo de capital esperado por estratégia")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "capital_por_estrategia.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 23. Gráfico — RoC por estratégia
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    resumo_estrategias["estrategia"],
    resumo_estrategias["roc_proxy"]
)

plt.ylabel("Retorno sobre capital — proxy")
plt.title("RoC proxy por estratégia")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "roc_por_estrategia.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 24. Gráfico — trade-off Retenção × Resultado
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    resumo_estrategias["retencao_media"],
    resumo_estrategias["resultado_esperado_total"],
    s=100
)

for _, linha in resumo_estrategias.iterrows():
    plt.annotate(
        linha["estrategia"],
        (
            linha["retencao_media"],
            linha["resultado_esperado_total"]
        )
    )

plt.xlabel("Retenção média prevista")
plt.ylabel("Resultado econômico esperado (R$)")
plt.title("Trade-off: retenção × resultado")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "tradeoff_retencao_resultado.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 25. Gráfico — Trade-off Capital × Resultado
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    resumo_estrategias["capital_proxy_total"],
    resumo_estrategias["resultado_esperado_total"],
    s=100
)

for _, linha in resumo_estrategias.iterrows():
    plt.annotate(
        linha["estrategia"],
        (
            linha["capital_proxy_total"],
            linha["resultado_esperado_total"]
        )
    )

plt.xlabel("Capital proxy esperado (R$)")
plt.ylabel("Resultado econômico esperado (R$)")
plt.title("Trade-off: capital × resultado")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "tradeoff_capital_resultado.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 26. Encerramento
# ------------------------------------------------------------

print("\nEtapa 9 concluída com sucesso.")
