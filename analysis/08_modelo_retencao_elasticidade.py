import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss
)

# ============================================================
# ETAPA 8 — MODELO DE RETENÇÃO / ELASTICIDADE AO PREÇO
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Introduzir o comportamento do cliente no motor de pricing.
#
# Nesta etapa:
# 1. reconstruímos o prêmio técnico/comercial das etapas anteriores;
# 2. criamos um experimento SINTÉTICO de renovação;
# 3. simulamos a decisão de retenção frente a diferentes preços;
# 4. ajustamos um GLM Binomial com link logit;
# 5. validamos o modelo fora da amostra;
# 6. construímos uma curva preço × retenção;
# 7. estimamos a elasticidade local da retenção ao preço.
#
# IMPORTANTE:
# A resposta comportamental é inteiramente sintética e didática.
# Não representa elasticidades reais do mercado brasileiro.
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
# 2. Parâmetros técnicos/comerciais herdados da Etapa 7
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
    raise ValueError(
        "Os carregamentos percentuais tornam o denominador comercial inválido."
    )


# ------------------------------------------------------------
# 3. Parâmetros do gerador comportamental sintético
# ------------------------------------------------------------
#
# O experimento simula uma oferta de renovação para cada apólice,
# variando o preço ao redor do prêmio vigente.
#
# Modelo verdadeiro:
#
# logit(p_i) =
#     beta_0
#     + beta_preco * log(P_oferta / P_vigente)
#     + beta_bonus * (bonus - 5)
#     + beta_comercial * I(uso = Comercial)
#     + beta_completa * I(cobertura = Completa)
#
# beta_preco < 0:
# aumentos de preço reduzem a probabilidade de retenção.

SEMENTE = 20260907
rng = np.random.default_rng(SEMENTE)

BETA0_RETENCAO = np.log(0.82 / (1 - 0.82))
BETA_PRECO = -4.00
BETA_BONUS = 0.04
BETA_COMERCIAL = -0.10
BETA_COMPLETA = 0.08

# Amplitude do experimento de preço.
# O fator é limitado para evitar propostas excessivamente extremas.
MEDIA_LOG_FATOR_TESTE = 0.00
DESVIO_LOG_FATOR_TESTE = 0.10
FATOR_MIN = 0.80
FATOR_MAX = 1.25


# ------------------------------------------------------------
# 4. Leitura das bases
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
# 5. Integridade
# ------------------------------------------------------------

if apolices.empty:
    raise ValueError("A base de apólices está vazia.")

if sinistros.empty:
    raise ValueError("A base de sinistros está vazia.")

if (apolices["exposicao"] <= 0).any():
    raise ValueError("Existem exposições menores ou iguais a zero.")

if (apolices["premio_vigente"] <= 0).any():
    raise ValueError("Existem prêmios vigentes menores ou iguais a zero.")

if (sinistros["valor_sinistro"] <= 0).any():
    raise ValueError("Existem valores de sinistro menores ou iguais a zero.")


# ------------------------------------------------------------
# 6. Reconstrução do GLM de frequência
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
# 7. Reconstrução do GLM de severidade
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
# 8. Reconstrução do prêmio puro e comercial indicado
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
    base["premio_puro_anual"] + DESPESA_FIXA_ANUAL
) / DENOMINADOR_COMERCIAL


# ------------------------------------------------------------
# 9. Construção do experimento sintético de preço
# ------------------------------------------------------------
#
# Cada apólice recebe uma oferta experimental em torno do preço vigente.
# O log da razão de preço é usado porque:
#
#     log(P_oferta / P_vigente)
#
# é simétrico em termos relativos e possui interpretação econômica natural.

fator_teste_preco = rng.lognormal(
    mean=MEDIA_LOG_FATOR_TESTE,
    sigma=DESVIO_LOG_FATOR_TESTE,
    size=len(base)
)

fator_teste_preco = np.clip(
    fator_teste_preco,
    FATOR_MIN,
    FATOR_MAX
)

base["fator_teste_preco"] = fator_teste_preco

base["premio_oferta_teste"] = (
    base["premio_vigente"]
    *
    base["fator_teste_preco"]
)

base["variacao_preco_teste_pct"] = (
    base["premio_oferta_teste"]
    /
    base["premio_vigente"]
    - 1
)

base["log_relatividade_preco"] = np.log(
    base["premio_oferta_teste"]
    /
    base["premio_vigente"]
)


# ------------------------------------------------------------
# 10. Probabilidade verdadeira de retenção
# ------------------------------------------------------------

indicador_comercial = (
    base["tipo_uso"].astype(str) == "Comercial"
).astype(int)

indicador_completa = (
    base["nivel_cobertura"].astype(str) == "Completa"
).astype(int)

eta_retencao = (
    BETA0_RETENCAO
    + BETA_PRECO * base["log_relatividade_preco"]
    + BETA_BONUS * (base["classe_bonus"] - 5)
    + BETA_COMERCIAL * indicador_comercial
    + BETA_COMPLETA * indicador_completa
)

base["prob_retencao_verdadeira"] = (
    1 / (1 + np.exp(-eta_retencao))
)

base["retido"] = rng.binomial(
    n=1,
    p=base["prob_retencao_verdadeira"]
)


# ------------------------------------------------------------
# 11. Resumo do experimento
# ------------------------------------------------------------

print("\n================ EXPERIMENTO SINTÉTICO DE RETENÇÃO ================")
print(f"Apólices: {len(base):,}")
print(f"Taxa de retenção observada: {base['retido'].mean():.2%}")
print(
    f"Probabilidade média verdadeira: "
    f"{base['prob_retencao_verdadeira'].mean():.2%}"
)
print(
    f"Variação média de preço no teste: "
    f"{base['variacao_preco_teste_pct'].mean():.2%}"
)
print(
    f"Variação mínima de preço: "
    f"{base['variacao_preco_teste_pct'].min():.2%}"
)
print(
    f"Variação máxima de preço: "
    f"{base['variacao_preco_teste_pct'].max():.2%}"
)


# ------------------------------------------------------------
# 12. Categorias do modelo de retenção
# ------------------------------------------------------------

base["tipo_uso_retencao"] = pd.Categorical(
    base["tipo_uso"].astype(str),
    categories=["Particular", "Comercial"]
)

base["nivel_cobertura_retencao"] = pd.Categorical(
    base["nivel_cobertura"].astype(str),
    categories=["Intermediaria", "Basica", "Completa"]
)


# ------------------------------------------------------------
# 13. Fórmula do GLM Binomial
# ------------------------------------------------------------
#
# logit(p_i) =
#     beta_0
#     + beta_1 * log_relatividade_preco
#     + beta_2 * classe_bonus
#     + efeitos de uso
#     + efeitos de cobertura

formula_retencao = """
retido
~
log_relatividade_preco
+
classe_bonus
+
C(tipo_uso_retencao, Treatment(reference="Particular"))
+
C(nivel_cobertura_retencao, Treatment(reference="Intermediaria"))
"""


# ------------------------------------------------------------
# 14. Modelo completo
# ------------------------------------------------------------

modelo_retencao = smf.glm(
    formula=formula_retencao,
    data=base,
    family=sm.families.Binomial()
).fit()

print("\n================ RESUMO DO GLM BINOMIAL ================")
print(modelo_retencao.summary())


# ------------------------------------------------------------
# 15. Coeficientes e Odds Ratios
# ------------------------------------------------------------

intervalos = modelo_retencao.conf_int()

tabela_coeficientes = pd.DataFrame({
    "coeficiente": modelo_retencao.params,
    "erro_padrao": modelo_retencao.bse,
    "p_valor": modelo_retencao.pvalues,
    "odds_ratio": np.exp(modelo_retencao.params),
    "OR_limite_inferior": np.exp(intervalos[0]),
    "OR_limite_superior": np.exp(intervalos[1])
})

print("\n================ COEFICIENTES E ODDS RATIOS ================")
print(tabela_coeficientes)

tabela_coeficientes.to_csv(
    PASTA_TABELAS / "coeficientes_glm_retencao.csv",
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16. Verdadeiro × estimado
# ------------------------------------------------------------
#
# O intercepto do modelo estimado usa classe_bonus sem centralização.
# No gerador:
#
# beta0 + beta_bonus * (bonus - 5)
# = (beta0 - 5 * beta_bonus) + beta_bonus * bonus

INTERCEPTO_EQUIVALENTE = (
    BETA0_RETENCAO - 5 * BETA_BONUS
)

coeficientes_verdadeiros = {
    "Intercept": INTERCEPTO_EQUIVALENTE,
    "log_relatividade_preco": BETA_PRECO,
    "classe_bonus": BETA_BONUS,
    'C(tipo_uso_retencao, Treatment(reference="Particular"))[T.Comercial]':
        BETA_COMERCIAL,
    'C(nivel_cobertura_retencao, Treatment(reference="Intermediaria"))[T.Basica]':
        0.0,
    'C(nivel_cobertura_retencao, Treatment(reference="Intermediaria"))[T.Completa]':
        BETA_COMPLETA
}

linhas = []

for termo, beta_verdadeiro in coeficientes_verdadeiros.items():
    beta_estimado = modelo_retencao.params.get(termo, np.nan)

    linhas.append({
        "termo": termo,
        "beta_verdadeiro": beta_verdadeiro,
        "beta_estimado": beta_estimado,
        "erro_estimacao": (
            beta_estimado - beta_verdadeiro
            if pd.notna(beta_estimado)
            else np.nan
        ),
        "odds_ratio_verdadeiro": np.exp(beta_verdadeiro),
        "odds_ratio_estimado": (
            np.exp(beta_estimado)
            if pd.notna(beta_estimado)
            else np.nan
        )
    })

comparacao_coeficientes = pd.DataFrame(linhas)

print("\n================ VERDADEIRO × ESTIMADO ================")
print(comparacao_coeficientes)

comparacao_coeficientes.to_csv(
    PASTA_TABELAS / "comparacao_coeficientes_retencao.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 17. Divisão treino/teste
# ------------------------------------------------------------

treino, teste = train_test_split(
    base,
    test_size=0.20,
    random_state=SEMENTE,
    stratify=base["retido"]
)

treino = treino.copy()
teste = teste.copy()

modelo_retencao_treino = smf.glm(
    formula=formula_retencao,
    data=treino,
    family=sm.families.Binomial()
).fit()

teste["prob_retencao_prevista"] = (
    modelo_retencao_treino.predict(teste)
)


# ------------------------------------------------------------
# 18. Métricas fora da amostra
# ------------------------------------------------------------

taxa_obs_teste = teste["retido"].mean()
taxa_prev_teste = teste["prob_retencao_prevista"].mean()

oe_teste = (
    taxa_obs_teste
    /
    taxa_prev_teste
)

auc_teste = roc_auc_score(
    teste["retido"],
    teste["prob_retencao_prevista"]
)

brier_teste = brier_score_loss(
    teste["retido"],
    teste["prob_retencao_prevista"]
)

logloss_teste = log_loss(
    teste["retido"],
    teste["prob_retencao_prevista"]
)

print("\n================ VALIDAÇÃO FORA DA AMOSTRA ================")
print(f"Registros no treino: {len(treino):,}")
print(f"Registros no teste: {len(teste):,}")
print(f"Retenção observada no teste: {taxa_obs_teste:.2%}")
print(f"Retenção prevista média no teste: {taxa_prev_teste:.2%}")
print(f"O/E retenção: {oe_teste:.4f}")
print(f"AUC: {auc_teste:.4f}")
print(f"Brier Score: {brier_teste:.4f}")
print(f"Log Loss: {logloss_teste:.4f}")

pd.DataFrame({
    "metrica": [
        "registros_treino",
        "registros_teste",
        "retencao_observada_teste",
        "retencao_prevista_teste",
        "OE_retencao_teste",
        "AUC_teste",
        "Brier_teste",
        "LogLoss_teste"
    ],
    "valor": [
        len(treino),
        len(teste),
        taxa_obs_teste,
        taxa_prev_teste,
        oe_teste,
        auc_teste,
        brier_teste,
        logloss_teste
    ]
}).to_csv(
    PASTA_TABELAS / "validacao_teste_glm_retencao.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 19. Calibração por decis de probabilidade
# ------------------------------------------------------------

teste["decil_probabilidade"] = pd.qcut(
    teste["prob_retencao_prevista"],
    q=10,
    duplicates="drop"
)

calibracao_decis = (
    teste
    .groupby("decil_probabilidade", observed=False)
    .agg(
        apolices=("id_apolice", "count"),
        retencao_observada=("retido", "mean"),
        retencao_prevista=("prob_retencao_prevista", "mean")
    )
)

calibracao_decis["OE"] = (
    calibracao_decis["retencao_observada"]
    /
    calibracao_decis["retencao_prevista"]
)

print("\n================ CALIBRAÇÃO POR DECIL ================")
print(calibracao_decis)

calibracao_decis.to_csv(
    PASTA_TABELAS / "calibracao_retencao_decis.csv",
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 20. Validação por faixa de variação de preço
# ------------------------------------------------------------

teste["faixa_variacao_preco"] = pd.cut(
    teste["variacao_preco_teste_pct"],
    bins=[-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf],
    labels=[
        "Redução >= 10%",
        "Redução de 3% a 10%",
        "Próximo (+/-3%)",
        "Aumento de 3% a 10%",
        "Aumento >= 10%"
    ]
)

validacao_preco = (
    teste
    .groupby("faixa_variacao_preco", observed=False)
    .agg(
        apolices=("id_apolice", "count"),
        variacao_media_preco=("variacao_preco_teste_pct", "mean"),
        retencao_observada=("retido", "mean"),
        retencao_prevista=("prob_retencao_prevista", "mean")
    )
)

validacao_preco["OE"] = (
    validacao_preco["retencao_observada"]
    /
    validacao_preco["retencao_prevista"]
)

print("\n================ VALIDAÇÃO POR FAIXA DE PREÇO ================")
print(validacao_preco)

validacao_preco.to_csv(
    PASTA_TABELAS / "validacao_retencao_por_preco.csv",
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 21. Refit final na base completa
# ------------------------------------------------------------
#
# Após a validação, usamos novamente a base completa para obter
# o modelo que será levado às etapas seguintes.

modelo_retencao_final = smf.glm(
    formula=formula_retencao,
    data=base,
    family=sm.families.Binomial()
).fit()


# ------------------------------------------------------------
# 22. Probabilidade de retenção sob o preço comercial indicado
# ------------------------------------------------------------

base["log_relatividade_preco_indicado"] = np.log(
    base["premio_comercial_indicado"]
    /
    base["premio_vigente"]
)

base_cenario_indicado = base.copy()

base_cenario_indicado["log_relatividade_preco"] = (
    base_cenario_indicado["log_relatividade_preco_indicado"]
)

base["prob_retencao_preco_indicado"] = (
    modelo_retencao_final.predict(base_cenario_indicado)
)

print("\n================ RETENÇÃO SOB PREÇO COMERCIAL INDICADO ================")
print(
    f"Retenção média prevista: "
    f"{base['prob_retencao_preco_indicado'].mean():.2%}"
)


# ------------------------------------------------------------
# 23. Curva preço × retenção
# ------------------------------------------------------------
#
# Construímos uma grade de multiplicadores sobre o prêmio vigente.
# Esta curva será usada na Etapa 9 para comparar estratégias.

multiplicadores = np.round(
    np.arange(0.80, 1.301, 0.025),
    3
)

linhas_curva = []

for multiplicador in multiplicadores:
    cenario = base.copy()

    cenario["log_relatividade_preco"] = np.log(
        multiplicador
    )

    prob = modelo_retencao_final.predict(cenario)

    linhas_curva.append({
        "multiplicador_preco_vigente": multiplicador,
        "variacao_preco_pct": multiplicador - 1,
        "retencao_media_prevista": prob.mean(),
        "apolices_esperadas_retidas": prob.sum(),
        "premio_medio_cenario": (
            base["premio_vigente"] * multiplicador
        ).mean()
    })

curva_preco_retencao = pd.DataFrame(linhas_curva)

print("\n================ CURVA PREÇO × RETENÇÃO ================")
print(curva_preco_retencao)

curva_preco_retencao.to_csv(
    PASTA_TABELAS / "curva_preco_retencao.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 24. Elasticidade local da retenção ao preço
# ------------------------------------------------------------
#
# Se:
#
# logit(p) = ... + beta * log(P)
#
# então:
#
# dp / dlog(P) = beta * p * (1-p)
#
# e a elasticidade de p em relação ao preço é:
#
# dlog(p) / dlog(P) = beta * (1-p)
#
# Como beta_preco é negativo, a elasticidade tende a ser negativa.

beta_preco_estimado = (
    modelo_retencao_final.params["log_relatividade_preco"]
)

base["elasticidade_retencao_preco_indicado"] = (
    beta_preco_estimado
    *
    (
        1
        -
        base["prob_retencao_preco_indicado"]
    )
)

elasticidade_media = (
    base["elasticidade_retencao_preco_indicado"].mean()
)

print("\n================ ELASTICIDADE DE RETENÇÃO ================")
print(f"Beta de preço estimado: {beta_preco_estimado:.4f}")
print(
    f"Elasticidade média no preço indicado: "
    f"{elasticidade_media:.4f}"
)


# ------------------------------------------------------------
# 25. Resumo por segmento sob preço indicado
# ------------------------------------------------------------

base["faixa_etaria_analise"] = pd.cut(
    base["idade_condutor"],
    bins=[17, 24, 34, 44, 54, 64, 80],
    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    include_lowest=True
)

def resumo_retencao_segmento(df, coluna):
    resultado = (
        df
        .groupby(coluna, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            premio_vigente_medio=("premio_vigente", "mean"),
            premio_indicado_medio=("premio_comercial_indicado", "mean"),
            premio_puro_medio=("premio_puro_anual", "mean"),
            retencao_media_prevista=(
                "prob_retencao_preco_indicado",
                "mean"
            ),
            elasticidade_media=(
                "elasticidade_retencao_preco_indicado",
                "mean"
            )
        )
    )

    resultado["variacao_preco_indicada_pct"] = (
        resultado["premio_indicado_medio"]
        /
        resultado["premio_vigente_medio"]
        - 1
    )

    resultado["apolices_esperadas_retidas"] = (
        resultado["apolices"]
        *
        resultado["retencao_media_prevista"]
    )

    return resultado

resumo_regiao = resumo_retencao_segmento(base, "regiao")
resumo_tipo_uso = resumo_retencao_segmento(base, "tipo_uso")
resumo_tipo_veiculo = resumo_retencao_segmento(base, "tipo_veiculo")
resumo_cobertura = resumo_retencao_segmento(base, "nivel_cobertura")
resumo_bonus = resumo_retencao_segmento(base, "classe_bonus")
resumo_faixa_etaria = resumo_retencao_segmento(
    base,
    "faixa_etaria_analise"
)

for nome, tabela in {
    "regiao": resumo_regiao,
    "tipo_uso": resumo_tipo_uso,
    "tipo_veiculo": resumo_tipo_veiculo,
    "cobertura": resumo_cobertura,
    "bonus": resumo_bonus,
    "faixa_etaria": resumo_faixa_etaria
}.items():
    print(f"\n================ RETENÇÃO POR {nome.upper()} ================")
    print(tabela)

    tabela.to_csv(
        PASTA_TABELAS / f"retencao_por_{nome}.csv",
        encoding="utf-8-sig"
    )


# ------------------------------------------------------------
# 26. Exportação da base de retenção
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
    "fator_teste_preco",
    "premio_oferta_teste",
    "variacao_preco_teste_pct",
    "log_relatividade_preco",
    "prob_retencao_verdadeira",
    "retido",
    "log_relatividade_preco_indicado",
    "prob_retencao_preco_indicado",
    "elasticidade_retencao_preco_indicado"
]

base[colunas_exportacao].to_csv(
    PASTA_TABELAS / "base_modelagem_retencao.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 27. Exportação de parâmetros
# ------------------------------------------------------------

pd.DataFrame({
    "parametro": [
        "semente",
        "beta0_retencao",
        "beta_preco",
        "beta_bonus",
        "beta_comercial",
        "beta_completa",
        "media_log_fator_teste",
        "desvio_log_fator_teste",
        "fator_min",
        "fator_max"
    ],
    "valor": [
        SEMENTE,
        BETA0_RETENCAO,
        BETA_PRECO,
        BETA_BONUS,
        BETA_COMERCIAL,
        BETA_COMPLETA,
        MEDIA_LOG_FATOR_TESTE,
        DESVIO_LOG_FATOR_TESTE,
        FATOR_MIN,
        FATOR_MAX
    ]
}).to_csv(
    PASTA_TABELAS / "parametros_gerador_retencao.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 28. Gráfico — curva preço × retenção
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    curva_preco_retencao["variacao_preco_pct"] * 100,
    curva_preco_retencao["retencao_media_prevista"] * 100,
    marker="o"
)

plt.xlabel("Variação do preço vs vigente (%)")
plt.ylabel("Retenção média prevista (%)")
plt.title("Curva preço × retenção")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "curva_preco_retencao.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 29. Gráfico — calibração por decis
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    calibracao_decis["retencao_prevista"],
    calibracao_decis["retencao_observada"],
    s=60
)

limite_min = min(
    calibracao_decis["retencao_prevista"].min(),
    calibracao_decis["retencao_observada"].min()
)

limite_max = max(
    calibracao_decis["retencao_prevista"].max(),
    calibracao_decis["retencao_observada"].max()
)

plt.plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    linestyle="--"
)

plt.xlabel("Retenção prevista")
plt.ylabel("Retenção observada")
plt.title("Calibração do modelo de retenção por decis")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "calibracao_retencao_decis.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 30. Gráfico — retenção por faixa de preço no teste
# ------------------------------------------------------------

categorias = validacao_preco.index.astype(str)
x = np.arange(len(categorias))
largura = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - largura / 2,
    validacao_preco["retencao_observada"],
    width=largura,
    label="Observada"
)

plt.bar(
    x + largura / 2,
    validacao_preco["retencao_prevista"],
    width=largura,
    label="Prevista"
)

plt.xticks(x, categorias, rotation=45)
plt.ylabel("Taxa de retenção")
plt.title("Retenção observada × prevista por faixa de preço")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "retencao_observada_prevista_por_preco.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 31. Gráfico — prêmio indicado e retenção esperada
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    base["premio_comercial_indicado"],
    base["prob_retencao_preco_indicado"],
    alpha=0.15,
    s=10
)

plt.xlabel("Prêmio comercial indicado (R$)")
plt.ylabel("Probabilidade prevista de retenção")
plt.title("Prêmio indicado × retenção prevista")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "premio_indicado_vs_retencao.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 32. Encerramento
# ------------------------------------------------------------

print("\nEtapa 8 concluída com sucesso.")