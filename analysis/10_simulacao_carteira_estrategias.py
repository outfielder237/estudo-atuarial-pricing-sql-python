import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# ETAPA 10 — SIMULAÇÃO DA CARTEIRA POR ESTRATÉGIA
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Simular, para cada estratégia de pricing:
#
#   Growth
#   Margin
#   Capital-efficient
#
# a carteira anual futura, incluindo:
#
#   1. retenção de apólices;
#   2. quantidade de sinistros;
#   3. severidade dos sinistros;
#   4. perda agregada;
#   5. prêmio emitido/retido;
#   6. despesas operacionais;
#   7. resultado técnico;
#   8. Loss Ratio.
#
# A saída desta etapa é uma DISTRIBUIÇÃO de resultados, e não
# apenas um valor esperado.
#
# Isso prepara diretamente a Etapa 11, em que a distribuição
# agregada de perdas será usada para medir capital econômico.
#
# IMPORTANTE
# - Carteira, retenção e parâmetros são sintéticos/didáticos.
# - Horizonte prospectivo: 1 ano.
# - Uma apólice retida é assumida com exposição anual = 1.
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

SEMENTE_SIMULACAO = 20260910

# 1.000 simulações fornece uma boa primeira distribuição
# sem tornar o script excessivamente pesado.
N_SIMULACOES = 1000

# Processamento em blocos reduz uso de memória.
TAMANHO_BLOCO = 25

# Shape da Gamma utilizado no gerador original de severidade.
GAMMA_SHAPE = 2.2


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
# 4. Parâmetros do modelo comportamental sintético
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
# 5. Parâmetros estratégicos — mesmos da Etapa 9
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
# 7. Integridade
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
# 8. GLM de frequência — especificação validada
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
# 9. GLM de severidade — especificação validada
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

# Como o horizonte prospectivo é anual, queremos lambda anual.
# Calculamos a previsão no período histórico e retiramos o efeito
# da exposição observada.
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
# 11. Proxy de intensidade de capital — herdada da Etapa 9
# ------------------------------------------------------------

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
# 13. Probabilidade de retenção por estratégia
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
# 14. Preparação dos vetores NumPy
# ------------------------------------------------------------

lambda_anual = base["frequencia_prevista"].to_numpy(dtype=float)
mu_severidade = base["severidade_prevista"].to_numpy(dtype=float)

if np.any(lambda_anual <= 0):
    raise ValueError("Existem frequências previstas <= 0.")

if np.any(mu_severidade <= 0):
    raise ValueError("Existem severidades previstas <= 0.")

# Scale da Gamma:
# E[Y] = shape * scale = mu
gamma_scale = (
    mu_severidade
    /
    GAMMA_SHAPE
)


# ------------------------------------------------------------
# 15. Configuração das estratégias
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
# 16. Função de simulação de uma estratégia
# ------------------------------------------------------------
#
# Para cada simulação m:
#
# R_i ~ Bernoulli(p_i)
#
# N_i | R_i ~ Poisson(R_i * lambda_i)
#
# Se N_i > 0:
#
# L_i | N_i
# ~ Gamma(shape = N_i * k, scale = mu_i / k)
#
# Isso é possível porque a soma de N variáveis Gamma com
# mesmo scale também é Gamma.
#
# Receita:
#
# P_m = sum_i R_i * preco_i
#
# Despesas:
#
# D_m =
#   despesas percentuais * P_m
#   + despesa fixa * número de apólices retidas
#
# Resultado:
#
# U_m = P_m - D_m - L_m

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

        # ----------------------------------------------------
        # 16.1 Retenção
        # ----------------------------------------------------

        retidos = (
            rng.random(
                size=(n_bloco, n_apolices)
            )
            <
            prob_retencao
        )

        qtd_retidos = retidos.sum(axis=1)

        # Receita da carteira retida
        premio_total = (
            retidos @ preco
        )

        # ----------------------------------------------------
        # 16.2 Frequência
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 16.3 Severidade e perda agregada
        # ----------------------------------------------------
        #
        # Para evitar gerar severidade para milhões de células
        # sem sinistro, simulamos somente onde N_i > 0.

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

        # ----------------------------------------------------
        # 16.4 Despesas e resultado
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 16.5 Armazenamento
        # ----------------------------------------------------

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

        print(
            f"{nome_estrategia}: "
            f"{simulacao_inicial}/{N_SIMULACOES} simulações",
            end="\r"
        )

    print()

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
# 18. Resumo estatístico das distribuições
# ------------------------------------------------------------

def resumo_distribuicao_estrategia(df):
    linhas = []

    for estrategia, grupo in df.groupby("estrategia"):
        resultado = grupo["resultado_tecnico"]
        perdas = grupo["perda_agregada"]

        linhas.append({
            "estrategia": estrategia,

            "retencao_media": (
                grupo["taxa_retencao_realizada"].mean()
            ),

            "apolices_retidas_media": (
                grupo["apolices_retidas"].mean()
            ),

            "sinistros_media": (
                grupo["sinistros"].mean()
            ),

            "premio_total_medio": (
                grupo["premio_total"].mean()
            ),

            "perda_media": perdas.mean(),
            "perda_desvio_padrao": perdas.std(ddof=1),
            "perda_p50": perdas.quantile(0.50),
            "perda_p75": perdas.quantile(0.75),
            "perda_p90": perdas.quantile(0.90),
            "perda_p95": perdas.quantile(0.95),
            "perda_p99": perdas.quantile(0.99),
            "perda_max": perdas.max(),

            "resultado_medio": resultado.mean(),
            "resultado_desvio_padrao": resultado.std(ddof=1),
            "resultado_p01": resultado.quantile(0.01),
            "resultado_p05": resultado.quantile(0.05),
            "resultado_p50": resultado.quantile(0.50),
            "resultado_p95": resultado.quantile(0.95),
            "resultado_p99": resultado.quantile(0.99),

            "prob_resultado_negativo": (
                (resultado < 0).mean()
            ),

            "loss_ratio_medio": (
                grupo["loss_ratio"].mean()
            ),

            "expense_ratio_medio": (
                grupo["expense_ratio"].mean()
            ),

            "combined_ratio_medio": (
                grupo["combined_ratio"].mean()
            )
        })

    return pd.DataFrame(linhas)

resumo_simulacao = resumo_distribuicao_estrategia(
    simulacoes_total
)

print("\n================ RESUMO DAS SIMULAÇÕES ================")
print(resumo_simulacao)


# ------------------------------------------------------------
# 19. Comparação simulado × esperado
# ------------------------------------------------------------
#
# Validamos se a média das simulações converge aos valores
# esperados analiticamente.

linhas_esperado = []

for nome, config in estrategias.items():
    preco = config["preco"]
    p = config["retencao"]

    apolices_esperadas = p.sum()

    sinistros_esperados = (
        p
        *
        lambda_anual
    ).sum()

    perdas_esperadas = (
        p
        *
        lambda_anual
        *
        mu_severidade
    ).sum()

    premio_esperado = (
        p
        *
        preco
    ).sum()

    despesas_esperadas = (
        premio_esperado
        *
        CUSTOS_PERCENTUAIS_OPERACIONAIS
        +
        DESPESA_FIXA_ANUAL
        *
        apolices_esperadas
    )

    resultado_esperado = (
        premio_esperado
        -
        despesas_esperadas
        -
        perdas_esperadas
    )

    linhas_esperado.append({
        "estrategia": nome,
        "apolices_retidas_esperado": apolices_esperadas,
        "sinistros_esperado": sinistros_esperados,
        "premio_esperado": premio_esperado,
        "perda_esperada": perdas_esperadas,
        "despesas_esperadas": despesas_esperadas,
        "resultado_esperado": resultado_esperado
    })

valores_esperados = pd.DataFrame(
    linhas_esperado
)

comparacao_media = resumo_simulacao.merge(
    valores_esperados,
    on="estrategia",
    how="left"
)

comparacao_media["erro_relativo_perda"] = (
    comparacao_media["perda_media"]
    /
    comparacao_media["perda_esperada"]
    - 1
)

comparacao_media["erro_relativo_resultado"] = (
    comparacao_media["resultado_medio"]
    /
    comparacao_media["resultado_esperado"]
    - 1
)

print("\n================ SIMULADO × ESPERADO ================")
print(
    comparacao_media[
        [
            "estrategia",
            "perda_media",
            "perda_esperada",
            "erro_relativo_perda",
            "resultado_medio",
            "resultado_esperado",
            "erro_relativo_resultado"
        ]
    ]
)


# ------------------------------------------------------------
# 20. Composição esperada do risco retido
# ------------------------------------------------------------
#
# Mostra como cada estratégia seleciona uma carteira diferente.

linhas_mix = []

for nome, config in estrategias.items():
    p = config["retencao"]

    peso = p / p.sum()

    linhas_mix.append({
        "estrategia": nome,
        "frequencia_media_retida": np.sum(
            peso * lambda_anual
        ),
        "severidade_media_retida": np.sum(
            peso * mu_severidade
        ),
        "premio_puro_medio_retido": np.sum(
            peso * base["premio_puro_anual"].to_numpy()
        ),
        "indice_capital_medio_retido": np.sum(
            peso * base["indice_capital_relativo"].to_numpy()
        )
    })

mix_risco = pd.DataFrame(
    linhas_mix
)

print("\n================ MIX DE RISCO RETIDO ================")
print(mix_risco)


# ------------------------------------------------------------
# 21. Exportação das tabelas
# ------------------------------------------------------------

simulacoes_total.to_csv(
    PASTA_TABELAS / "simulacao_carteira_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

resumo_simulacao.to_csv(
    PASTA_TABELAS / "resumo_simulacao_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

comparacao_media.to_csv(
    PASTA_TABELAS / "simulado_vs_esperado_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

mix_risco.to_csv(
    PASTA_TABELAS / "mix_risco_retido_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "parametro": [
        "semente_simulacao",
        "n_simulacoes",
        "tamanho_bloco",
        "gamma_shape",
        "despesa_fixa_anual",
        "custos_percentuais_operacionais",
        "fator_growth",
        "fator_margin",
        "fator_capital_min",
        "fator_capital_max"
    ],
    "valor": [
        SEMENTE_SIMULACAO,
        N_SIMULACOES,
        TAMANHO_BLOCO,
        GAMMA_SHAPE,
        DESPESA_FIXA_ANUAL,
        CUSTOS_PERCENTUAIS_OPERACIONAIS,
        FATOR_GROWTH,
        FATOR_MARGIN,
        FATOR_CAPITAL_MIN,
        FATOR_CAPITAL_MAX
    ]
}).to_csv(
    PASTA_TABELAS / "parametros_simulacao_etapa10.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 22. Gráfico — distribuição da perda agregada
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for estrategia, grupo in simulacoes_total.groupby("estrategia"):
    plt.hist(
        grupo["perda_agregada"],
        bins=40,
        alpha=0.45,
        label=estrategia
    )

plt.xlabel("Perda agregada anual (R$)")
plt.ylabel("Frequência das simulações")
plt.title("Distribuição da perda agregada por estratégia")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "distribuicao_perda_agregada_estrategias.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 23. Gráfico — distribuição do resultado técnico
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for estrategia, grupo in simulacoes_total.groupby("estrategia"):
    plt.hist(
        grupo["resultado_tecnico"],
        bins=40,
        alpha=0.45,
        label=estrategia
    )

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Resultado técnico anual (R$)")
plt.ylabel("Frequência das simulações")
plt.title("Distribuição do resultado técnico por estratégia")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "distribuicao_resultado_estrategias.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 24. Gráfico — Boxplot das perdas
# ------------------------------------------------------------

ordem = [
    "Growth",
    "Capital-efficient",
    "Margin"
]

dados_boxplot_perda = [
    simulacoes_total.loc[
        simulacoes_total["estrategia"] == estrategia,
        "perda_agregada"
    ].to_numpy()
    for estrategia in ordem
]

plt.figure(figsize=(9, 5))

plt.boxplot(
    dados_boxplot_perda,
    tick_labels=ordem,
    showfliers=False
)

plt.ylabel("Perda agregada anual (R$)")
plt.title("Distribuição da perda agregada — comparação")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "boxplot_perdas_estrategias.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 25. Gráfico — Resultado médio e P5
# ------------------------------------------------------------

resumo_plot = resumo_simulacao.set_index("estrategia").loc[
    ordem
]

x = np.arange(len(ordem))
largura = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - largura / 2,
    resumo_plot["resultado_medio"],
    width=largura,
    label="Média"
)

plt.bar(
    x + largura / 2,
    resumo_plot["resultado_p05"],
    width=largura,
    label="P5"
)

plt.xticks(
    x,
    ordem,
    rotation=15
)

plt.ylabel("Resultado técnico (R$)")
plt.title("Resultado médio × cenário adverso P5")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "resultado_medio_vs_p05_estrategias.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 26. Gráfico — Probabilidade de resultado negativo
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    resumo_plot.index,
    resumo_plot["prob_resultado_negativo"]
)

plt.ylabel("Probabilidade")
plt.title("Probabilidade de resultado técnico negativo")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "probabilidade_resultado_negativo.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 27. Encerramento
# ------------------------------------------------------------

print("\nEtapa 10 concluída com sucesso.")