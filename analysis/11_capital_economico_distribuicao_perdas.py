import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# ETAPA 11 — CAPITAL ECONÔMICO A PARTIR DA DISTRIBUIÇÃO DE PERDAS
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Substituir a proxy de capital usada nas Etapas 9 e 10 por
# medidas de capital econômico calculadas diretamente a partir
# da distribuição simulada de perdas e resultados.
#
# Para cada estratégia:
#   Growth
#   Margin
#   Capital-efficient
#
# calculamos:
#
#   - perda média
#   - VaR da perda
#   - TVaR / Expected Shortfall da perda
#   - capital econômico baseado em VaR
#   - capital econômico baseado em TVaR
#   - déficit técnico em cenários adversos
#   - capital baseado na distribuição do resultado
#   - retorno sobre capital econômico
#
# Convenções principais:
#
#   CE_VaR(alpha) = VaR_alpha(L) - E[L]
#
#   CE_TVaR(alpha) = TVaR_alpha(L) - E[L]
#
#   onde L é a perda agregada anual.
#
# Também calculamos uma medida equivalente a partir do resultado:
#
#   Capital_Resultado(alpha)
#   = E[Resultado] - Quantil_(1-alpha)(Resultado)
#
# IMPORTANTE
# - Todos os dados são sintéticos/didáticos.
# - Horizonte prospectivo: 1 ano.
# - O nível central de capital é 99,5%, alinhado ao padrão
#   conceitual de solvência de 1 em 200 anos.
# - Como estimar 99,5% com apenas 1.000 simulações é instável,
#   esta etapa aumenta o número de simulações para 10.000.
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
# 2. Parâmetros de simulação
# ------------------------------------------------------------

SEMENTE_SIMULACAO = 20260911

# 10.000 simulações:
# melhora bastante a leitura da cauda de 99,5% em relação
# às 1.000 simulações da Etapa 10.
N_SIMULACOES = 10000

# Tamanho de bloco para controlar memória.
TAMANHO_BLOCO = 20

# Shape da Gamma do gerador original.
GAMMA_SHAPE = 2.2

# Níveis de confiança analisados.
NIVEIS_ALPHA = [0.95, 0.99, 0.995]


# ------------------------------------------------------------
# 3. Parâmetros comerciais
# ------------------------------------------------------------

DESPESA_FIXA_ANUAL = 40.00
COMISSAO_PCT = 0.12
TRIBUTOS_ENCARGOS_PCT = 0.0765
DESPESA_VARIAVEL_PCT = 0.08
MARGEM_ALVO_PCT = 0.05

CUSTOS_PERCENTUAIS_OPERACIONAIS = (
    COMISSAO_PCT
    + TRIBUTOS_ENCARGOS_PCT
    + DESPESA_VARIAVEL_PCT
)

SOMA_CARREGAMENTOS_PCT = (
    CUSTOS_PERCENTUAIS_OPERACIONAIS
    + MARGEM_ALVO_PCT
)

DENOMINADOR_COMERCIAL = 1 - SOMA_CARREGAMENTOS_PCT

if DENOMINADOR_COMERCIAL <= 0:
    raise ValueError(
        "Carregamentos percentuais tornam o prêmio comercial inválido."
    )


# ------------------------------------------------------------
# 4. Parâmetros comportamentais sintéticos
# ------------------------------------------------------------

BETA0_RETENCAO = np.log(0.82 / (1 - 0.82))
BETA_PRECO = -4.00
BETA_BONUS = 0.04
BETA_COMERCIAL = -0.10
BETA_COMPLETA = 0.08

INTERCEPTO_EQUIVALENTE = (
    BETA0_RETENCAO
    - 5 * BETA_BONUS
)


# ------------------------------------------------------------
# 5. Parâmetros estratégicos
# ------------------------------------------------------------

FATOR_GROWTH = 0.95
FATOR_MARGIN = 1.10

CARGA_CAPITAL_BASE = 0.20
PESO_SEVERIDADE_CAPITAL = 0.50
PESO_FREQUENCIA_CAPITAL = 0.50

FATOR_CAPITAL_MIN = 0.90
FATOR_CAPITAL_MAX = 1.20


# ------------------------------------------------------------
# 6. Leitura das bases
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
# 7. Testes de integridade
# ------------------------------------------------------------

if apolices.empty:
    raise ValueError("A base de apólices está vazia.")

if sinistros.empty:
    raise ValueError("A base de sinistros está vazia.")

if (apolices["exposicao"] <= 0).any():
    raise ValueError("Existem exposições <= 0.")

if (apolices["premio_vigente"] <= 0).any():
    raise ValueError("Existem prêmios vigentes <= 0.")

if (sinistros["valor_sinistro"] <= 0).any():
    raise ValueError("Existem valores de sinistro <= 0.")


# ------------------------------------------------------------
# 8. GLM de frequência
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
# 9. GLM de severidade
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
# 10. Base técnica prospectiva
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

base["premio_comercial_indicado"] = (
    base["premio_puro_anual"]
    + DESPESA_FIXA_ANUAL
) / DENOMINADOR_COMERCIAL


# ------------------------------------------------------------
# 11. Intensidade de capital usada apenas para definir
#     a estratégia Capital-efficient
# ------------------------------------------------------------
#
# Importante:
# esta proxy NÃO será a medida final de capital econômico.
# Ela é mantida apenas porque a própria estratégia
# Capital-efficient foi definida dessa forma na Etapa 9.

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
# 12. Preços das estratégias
# ------------------------------------------------------------

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
# 13. Retenção por estratégia
# ------------------------------------------------------------

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
# 14. Vetores NumPy
# ------------------------------------------------------------

lambda_anual = base["frequencia_prevista"].to_numpy(dtype=float)
mu_severidade = base["severidade_prevista"].to_numpy(dtype=float)

gamma_scale = (
    mu_severidade
    /
    GAMMA_SHAPE
)

if np.any(lambda_anual <= 0):
    raise ValueError("Existem frequências previstas <= 0.")

if np.any(mu_severidade <= 0):
    raise ValueError("Existem severidades previstas <= 0.")


# ------------------------------------------------------------
# 15. Estratégias
# ------------------------------------------------------------

estrategias = {
    "Growth": {
        "preco": base["preco_growth"].to_numpy(dtype=float),
        "retencao": base["retencao_growth"].to_numpy(dtype=float)
    },
    "Margin": {
        "preco": base["preco_margin"].to_numpy(dtype=float),
        "retencao": base["retencao_margin"].to_numpy(dtype=float)
    },
    "Capital-efficient": {
        "preco": base["preco_capital_efficient"].to_numpy(dtype=float),
        "retencao": base["retencao_capital_efficient"].to_numpy(dtype=float)
    }
}


# ------------------------------------------------------------
# 16. Função de simulação
# ------------------------------------------------------------
#
# A lógica é a mesma da Etapa 10, mas agora com maior número
# de simulações para permitir análise de cauda.

def simular_estrategia(
    nome_estrategia,
    preco,
    prob_retencao,
    seed
):
    rng = np.random.default_rng(seed)

    resultados = []

    n_apolices = len(base)
    simulacao_inicial = 0

    while simulacao_inicial < N_SIMULACOES:
        n_bloco = min(
            TAMANHO_BLOCO,
            N_SIMULACOES - simulacao_inicial
        )

        # Retenção
        retidos = (
            rng.random(
                size=(n_bloco, n_apolices)
            )
            <
            prob_retencao
        )

        qtd_retidos = retidos.sum(axis=1)

        # Receita
        premio_total = (
            retidos @ preco
        )

        # Frequência
        media_poisson = (
            retidos
            *
            lambda_anual
        )

        qtd_sinistros_matriz = rng.poisson(
            media_poisson
        )

        qtd_sinistros_total = (
            qtd_sinistros_matriz.sum(axis=1)
        )

        # Severidade
        linhas_pos, colunas_pos = np.nonzero(
            qtd_sinistros_matriz
        )

        perdas_total = np.zeros(
            n_bloco,
            dtype=float
        )

        if len(linhas_pos) > 0:
            n_claims_pos = (
                qtd_sinistros_matriz[
                    linhas_pos,
                    colunas_pos
                ]
            )

            shapes_agregados = (
                n_claims_pos
                *
                GAMMA_SHAPE
            )

            scales_agregados = (
                gamma_scale[
                    colunas_pos
                ]
            )

            perdas_por_risco = rng.gamma(
                shape=shapes_agregados,
                scale=scales_agregados
            )

            perdas_total = np.bincount(
                linhas_pos,
                weights=perdas_por_risco,
                minlength=n_bloco
            )

        # Despesas
        custos_percentuais = (
            premio_total
            *
            CUSTOS_PERCENTUAIS_OPERACIONAIS
        )

        despesa_fixa_total = (
            qtd_retidos
            *
            DESPESA_FIXA_ANUAL
        )

        despesas_operacionais = (
            custos_percentuais
            +
            despesa_fixa_total
        )

        # Resultado
        resultado_tecnico = (
            premio_total
            -
            despesas_operacionais
            -
            perdas_total
        )

        loss_ratio = np.divide(
            perdas_total,
            premio_total,
            out=np.full(
                n_bloco,
                np.nan,
                dtype=float
            ),
            where=premio_total > 0
        )

        expense_ratio = np.divide(
            despesas_operacionais,
            premio_total,
            out=np.full(
                n_bloco,
                np.nan,
                dtype=float
            ),
            where=premio_total > 0
        )

        combined_ratio = (
            loss_ratio
            +
            expense_ratio
        )

        for j in range(n_bloco):
            resultados.append({
                "estrategia": nome_estrategia,
                "simulacao": simulacao_inicial + j + 1,
                "apolices_retidas": int(qtd_retidos[j]),
                "taxa_retencao_realizada": (
                    qtd_retidos[j] / n_apolices
                ),
                "sinistros": int(
                    qtd_sinistros_total[j]
                ),
                "premio_total": premio_total[j],
                "perda_agregada": perdas_total[j],
                "despesas_operacionais": (
                    despesas_operacionais[j]
                ),
                "resultado_tecnico": (
                    resultado_tecnico[j]
                ),
                "loss_ratio": loss_ratio[j],
                "expense_ratio": expense_ratio[j],
                "combined_ratio": combined_ratio[j]
            })

        simulacao_inicial += n_bloco

        if (
            simulacao_inicial % 500 == 0
            or simulacao_inicial == N_SIMULACOES
        ):
            print(
                f"{nome_estrategia}: "
                f"{simulacao_inicial}/{N_SIMULACOES} simulações"
            )

    return pd.DataFrame(resultados)


# ------------------------------------------------------------
# 17. Execução das simulações
# ------------------------------------------------------------

print("\n================ INÍCIO DAS SIMULAÇÕES ================")

simulacoes = []

for indice, (nome, config) in enumerate(
    estrategias.items()
):
    tabela = simular_estrategia(
        nome_estrategia=nome,
        preco=config["preco"],
        prob_retencao=config["retencao"],
        seed=SEMENTE_SIMULACAO + indice * 1000
    )

    simulacoes.append(tabela)

simulacoes_total = pd.concat(
    simulacoes,
    ignore_index=True
)

print("\nSimulações concluídas.")


# ------------------------------------------------------------
# 18. Funções de risco
# ------------------------------------------------------------

def calcular_var(serie, alpha):
    """
    VaR alpha da perda.
    """
    return serie.quantile(alpha)


def calcular_tvar(serie, alpha):
    """
    TVaR / Expected Shortfall alpha:
    média das perdas acima ou iguais ao VaR alpha.
    """
    var_alpha = calcular_var(
        serie,
        alpha
    )

    cauda = serie[
        serie >= var_alpha
    ]

    return cauda.mean()


# ------------------------------------------------------------
# 19. Capital econômico por estratégia
# ------------------------------------------------------------

linhas_capital = []

for estrategia, grupo in (
    simulacoes_total.groupby("estrategia")
):
    perdas = grupo["perda_agregada"]
    resultado = grupo["resultado_tecnico"]

    perda_media = perdas.mean()
    resultado_medio = resultado.mean()

    for alpha in NIVEIS_ALPHA:
        var_perda = calcular_var(
            perdas,
            alpha
        )

        tvar_perda = calcular_tvar(
            perdas,
            alpha
        )

        capital_var = (
            var_perda
            -
            perda_media
        )

        capital_tvar = (
            tvar_perda
            -
            perda_media
        )

        # Capital equivalente pela distribuição do resultado:
        # pior quantil de resultado correspondente ao mesmo alpha.
        quantil_resultado_adverso = (
            resultado.quantile(
                1 - alpha
            )
        )

        capital_resultado = (
            resultado_medio
            -
            quantil_resultado_adverso
        )

        linhas_capital.append({
            "estrategia": estrategia,
            "alpha": alpha,
            "perda_media": perda_media,
            "VaR_perda": var_perda,
            "TVaR_perda": tvar_perda,
            "capital_economico_VaR": capital_var,
            "capital_economico_TVaR": capital_tvar,
            "resultado_medio": resultado_medio,
            "quantil_resultado_adverso": (
                quantil_resultado_adverso
            ),
            "capital_resultado": (
                capital_resultado
            )
        })

capital_economico = pd.DataFrame(
    linhas_capital
)

print("\n================ CAPITAL ECONÔMICO ================")
print(capital_economico)


# ------------------------------------------------------------
# 20. Medidas adicionais de cauda
# ------------------------------------------------------------

linhas_cauda = []

for estrategia, grupo in (
    simulacoes_total.groupby("estrategia")
):
    perda = grupo["perda_agregada"]
    resultado = grupo["resultado_tecnico"]
    combined = grupo["combined_ratio"]

    linhas_cauda.append({
        "estrategia": estrategia,
        "perda_media": perda.mean(),
        "perda_desvio_padrao": perda.std(ddof=1),
        "coeficiente_variacao_perda": (
            perda.std(ddof=1)
            /
            perda.mean()
        ),
        "resultado_medio": resultado.mean(),
        "resultado_desvio_padrao": (
            resultado.std(ddof=1)
        ),
        "prob_resultado_negativo": (
            resultado < 0
        ).mean(),
        "prob_combined_ratio_acima_100": (
            combined > 1
        ).mean(),
        "combined_ratio_p95": (
            combined.quantile(0.95)
        ),
        "combined_ratio_p99": (
            combined.quantile(0.99)
        ),
        "combined_ratio_p995": (
            combined.quantile(0.995)
        )
    })

resumo_cauda = pd.DataFrame(
    linhas_cauda
)

print("\n================ RESUMO DE CAUDA ================")
print(resumo_cauda)


# ------------------------------------------------------------
# 21. RoC com capital econômico real
# ------------------------------------------------------------

capital_995 = (
    capital_economico[
        capital_economico["alpha"] == 0.995
    ]
    .copy()
)

capital_995["RoC_VaR_995"] = (
    capital_995["resultado_medio"]
    /
    capital_995["capital_economico_VaR"].replace(
        0,
        np.nan
    )
)

capital_995["RoC_TVaR_995"] = (
    capital_995["resultado_medio"]
    /
    capital_995["capital_economico_TVaR"].replace(
        0,
        np.nan
    )
)

capital_995["RoC_resultado_995"] = (
    capital_995["resultado_medio"]
    /
    capital_995["capital_resultado"].replace(
        0,
        np.nan
    )
)

print("\n================ RoC — CAPITAL ECONÔMICO 99,5% ================")
print(
    capital_995[
        [
            "estrategia",
            "resultado_medio",
            "capital_economico_VaR",
            "capital_economico_TVaR",
            "capital_resultado",
            "RoC_VaR_995",
            "RoC_TVaR_995",
            "RoC_resultado_995"
        ]
    ]
)


# ------------------------------------------------------------
# 22. Ranking econômico final
# ------------------------------------------------------------

ranking = capital_995[
    [
        "estrategia",
        "resultado_medio",
        "capital_economico_VaR",
        "capital_economico_TVaR",
        "RoC_VaR_995",
        "RoC_TVaR_995"
    ]
].copy()

ranking["rank_resultado"] = (
    ranking["resultado_medio"]
    .rank(
        ascending=False,
        method="min"
    )
)

ranking["rank_menor_capital_var"] = (
    ranking["capital_economico_VaR"]
    .rank(
        ascending=True,
        method="min"
    )
)

ranking["rank_menor_capital_tvar"] = (
    ranking["capital_economico_TVaR"]
    .rank(
        ascending=True,
        method="min"
    )
)

ranking["rank_roc_var"] = (
    ranking["RoC_VaR_995"]
    .rank(
        ascending=False,
        method="min"
    )
)

ranking["rank_roc_tvar"] = (
    ranking["RoC_TVaR_995"]
    .rank(
        ascending=False,
        method="min"
    )
)

print("\n================ RANKING ECONÔMICO ================")
print(ranking)


# ------------------------------------------------------------
# 23. Sanity check: capital deve ser não-negativo
# ------------------------------------------------------------

colunas_capital = [
    "capital_economico_VaR",
    "capital_economico_TVaR",
    "capital_resultado"
]

for coluna in colunas_capital:
    if (
        capital_economico[coluna] < 0
    ).any():
        raise ValueError(
            f"Foram encontrados valores negativos em {coluna}."
        )


# ------------------------------------------------------------
# 24. Quantis completos de perda e resultado
# ------------------------------------------------------------

quantis = [
    0.01,
    0.05,
    0.50,
    0.90,
    0.95,
    0.99,
    0.995
]

linhas_quantis = []

for estrategia, grupo in (
    simulacoes_total.groupby("estrategia")
):
    for q in quantis:
        linhas_quantis.append({
            "estrategia": estrategia,
            "quantil": q,
            "perda_agregada": (
                grupo["perda_agregada"].quantile(q)
            ),
            "resultado_tecnico": (
                grupo["resultado_tecnico"].quantile(q)
            ),
            "combined_ratio": (
                grupo["combined_ratio"].quantile(q)
            )
        })

tabela_quantis = pd.DataFrame(
    linhas_quantis
)


# ------------------------------------------------------------
# 25. Exportações
# ------------------------------------------------------------

simulacoes_total.to_csv(
    PASTA_TABELAS / "simulacoes_etapa11_capital_economico.csv",
    index=False,
    encoding="utf-8-sig"
)

capital_economico.to_csv(
    PASTA_TABELAS / "capital_economico_por_estrategia.csv",
    index=False,
    encoding="utf-8-sig"
)

resumo_cauda.to_csv(
    PASTA_TABELAS / "resumo_cauda_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

capital_995.to_csv(
    PASTA_TABELAS / "capital_economico_995_roc.csv",
    index=False,
    encoding="utf-8-sig"
)

ranking.to_csv(
    PASTA_TABELAS / "ranking_capital_economico.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_quantis.to_csv(
    PASTA_TABELAS / "quantis_perda_resultado.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "parametro": [
        "semente_simulacao",
        "n_simulacoes",
        "tamanho_bloco",
        "gamma_shape",
        "alpha_principal",
        "despesa_fixa_anual",
        "custos_percentuais_operacionais"
    ],
    "valor": [
        SEMENTE_SIMULACAO,
        N_SIMULACOES,
        TAMANHO_BLOCO,
        GAMMA_SHAPE,
        0.995,
        DESPESA_FIXA_ANUAL,
        CUSTOS_PERCENTUAIS_OPERACIONAIS
    ]
}).to_csv(
    PASTA_TABELAS / "parametros_capital_economico_etapa11.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 26. Gráfico — distribuição da perda com VaR 99,5%
# ------------------------------------------------------------

for estrategia, grupo in (
    simulacoes_total.groupby("estrategia")
):
    perdas = grupo["perda_agregada"]

    var_995 = perdas.quantile(0.995)

    plt.figure(figsize=(10, 6))

    plt.hist(
        perdas,
        bins=50,
        alpha=0.75
    )

    plt.axvline(
        perdas.mean(),
        linestyle="--",
        label="Média"
    )

    plt.axvline(
        var_995,
        linestyle="--",
        label="VaR 99,5%"
    )

    plt.xlabel("Perda agregada anual (R$)")
    plt.ylabel("Frequência")
    plt.title(
        f"Distribuição da perda — {estrategia}"
    )
    plt.legend()
    plt.tight_layout()

    nome_arquivo = (
        estrategia
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        PASTA_FIGURAS
        / f"perda_var995_{nome_arquivo}.png",
        dpi=150
    )

    plt.show()


# ------------------------------------------------------------
# 27. Gráfico — Capital VaR e TVaR 99,5%
# ------------------------------------------------------------

capital_plot = (
    capital_995
    .set_index("estrategia")
)

ordem = [
    "Growth",
    "Capital-efficient",
    "Margin"
]

capital_plot = capital_plot.loc[
    ordem
]

x = np.arange(
    len(ordem)
)

largura = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - largura / 2,
    capital_plot["capital_economico_VaR"],
    width=largura,
    label="Capital VaR 99,5%"
)

plt.bar(
    x + largura / 2,
    capital_plot["capital_economico_TVaR"],
    width=largura,
    label="Capital TVaR 99,5%"
)

plt.xticks(
    x,
    ordem,
    rotation=15
)

plt.ylabel("Capital econômico (R$)")
plt.title("Capital econômico por estratégia")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "capital_economico_var_tvar_995.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 28. Gráfico — RoC econômico 99,5%
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    capital_plot.index,
    capital_plot["RoC_VaR_995"]
)

plt.ylabel("Retorno sobre capital econômico")
plt.title("RoC baseado em VaR 99,5%")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "roc_economico_var995.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 29. Gráfico — Resultado médio × Capital econômico
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    capital_995["capital_economico_VaR"],
    capital_995["resultado_medio"],
    s=100
)

for _, linha in capital_995.iterrows():
    plt.annotate(
        linha["estrategia"],
        (
            linha["capital_economico_VaR"],
            linha["resultado_medio"]
        )
    )

plt.xlabel("Capital econômico VaR 99,5% (R$)")
plt.ylabel("Resultado médio (R$)")
plt.title(
    "Trade-off: resultado × capital econômico"
)
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "tradeoff_resultado_capital_economico.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 30. Gráfico — Combined Ratio de cauda
# ------------------------------------------------------------

cauda_plot = (
    resumo_cauda
    .set_index("estrategia")
    .loc[ordem]
)

x = np.arange(
    len(ordem)
)

largura = 0.25

plt.figure(figsize=(10, 5))

plt.bar(
    x - largura,
    cauda_plot["combined_ratio_p95"],
    width=largura,
    label="P95"
)

plt.bar(
    x,
    cauda_plot["combined_ratio_p99"],
    width=largura,
    label="P99"
)

plt.bar(
    x + largura,
    cauda_plot["combined_ratio_p995"],
    width=largura,
    label="P99,5"
)

plt.axhline(
    1.0,
    linestyle="--",
    label="100%"
)

plt.xticks(
    x,
    ordem,
    rotation=15
)

plt.ylabel("Combined Ratio")
plt.title("Combined Ratio em cenários de cauda")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "combined_ratio_cauda.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 31. Encerramento
# ------------------------------------------------------------

print("\nEtapa 11 concluída com sucesso.")