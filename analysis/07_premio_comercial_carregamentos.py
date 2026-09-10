import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# ETAPA 7 — PRÊMIO COMERCIAL E CARREGAMENTOS
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Transformar o prêmio puro anual em prêmio comercial indicado,
# incorporando carregamentos explícitos e rastreáveis.
#
# Estrutura principal:
#
#                    PP_i + D_fixa
# PC_i = ------------------------------------------
#        1 - (c + t + d_var + m)
#
# onde:
# PP_i   = prêmio puro anual
# D_fixa = despesa fixa anual por apólice
# c      = comissão (% do prêmio comercial)
# t      = tributos/encargos (% do prêmio comercial)
# d_var  = despesas variáveis (% do prêmio comercial)
# m      = margem-alvo (% do prêmio comercial)
#
# IMPORTANTE:
# Todos os parâmetros desta etapa são DIDÁTICOS e SINTÉTICOS.
# Não representam necessariamente taxas, despesas ou margens
# observadas no mercado brasileiro.
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
# 2. Parâmetros comerciais — cenário-base
# ------------------------------------------------------------
#
# Estes valores são deliberadamente configuráveis.
# O objetivo é permitir que a etapa seja reproduzida e,
# posteriormente, submetida a análises de sensibilidade.

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

if SOMA_CARREGAMENTOS_PCT >= 1:
    raise ValueError(
        "A soma dos carregamentos percentuais deve ser menor que 100%."
    )

print("\n================ PARÂMETROS COMERCIAIS ================")
print(f"Despesa fixa anual por apólice: R$ {DESPESA_FIXA_ANUAL:,.2f}")
print(f"Comissão: {COMISSAO_PCT:.2%}")
print(f"Tributos/encargos: {TRIBUTOS_ENCARGOS_PCT:.2%}")
print(f"Despesa variável: {DESPESA_VARIAVEL_PCT:.2%}")
print(f"Margem-alvo: {MARGEM_ALVO_PCT:.2%}")
print(f"Soma dos carregamentos percentuais: {SOMA_CARREGAMENTOS_PCT:.2%}")


# ------------------------------------------------------------
# 3. Leitura da base no SQLite
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
# 4. Testes de integridade
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
# 5. Reajuste do GLM de frequência
# ------------------------------------------------------------
#
# Mantemos a mesma especificação validada na Etapa 4.

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
# 6. Reajuste do GLM de severidade
# ------------------------------------------------------------
#
# Mantemos a mesma especificação validada na Etapa 5.

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
# 7. Construção da base técnica de pricing
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


# ------------------------------------------------------------
# 8. Construção do prêmio comercial indicado
# ------------------------------------------------------------
#
# Equação:
#
#                    PP_i + D_fixa
# PC_i = ------------------------------------------
#        1 - (c + t + d_var + m)
#
# A razão para dividir, em vez de simplesmente somar percentuais
# sobre PP_i, é que esses componentes são tratados como percentuais
# do próprio prêmio comercial.

DENOMINADOR_COMERCIAL = 1 - SOMA_CARREGAMENTOS_PCT

base["premio_comercial_indicado"] = (
    base["premio_puro_anual"] + DESPESA_FIXA_ANUAL
) / DENOMINADOR_COMERCIAL


# ------------------------------------------------------------
# 9. Decomposição monetária do prêmio comercial
# ------------------------------------------------------------

base["comissao_valor"] = (
    base["premio_comercial_indicado"]
    *
    COMISSAO_PCT
)

base["tributos_encargos_valor"] = (
    base["premio_comercial_indicado"]
    *
    TRIBUTOS_ENCARGOS_PCT
)

base["despesa_variavel_valor"] = (
    base["premio_comercial_indicado"]
    *
    DESPESA_VARIAVEL_PCT
)

base["margem_alvo_valor"] = (
    base["premio_comercial_indicado"]
    *
    MARGEM_ALVO_PCT
)

base["despesa_fixa_valor"] = DESPESA_FIXA_ANUAL

base["total_carregamentos_valor"] = (
    base["comissao_valor"]
    + base["tributos_encargos_valor"]
    + base["despesa_variavel_valor"]
    + base["margem_alvo_valor"]
    + base["despesa_fixa_valor"]
)

# Checagem contábil da decomposição:
# prêmio comercial = prêmio puro + todos os carregamentos
base["diferenca_fechamento"] = (
    base["premio_comercial_indicado"]
    - (
        base["premio_puro_anual"]
        + base["comissao_valor"]
        + base["tributos_encargos_valor"]
        + base["despesa_variavel_valor"]
        + base["margem_alvo_valor"]
        + base["despesa_fixa_valor"]
    )
)

erro_fechamento_max = base["diferenca_fechamento"].abs().max()

if erro_fechamento_max > 1e-6:
    raise ValueError(
        f"Falha no fechamento do prêmio comercial. "
        f"Erro máximo: {erro_fechamento_max}"
    )


# ------------------------------------------------------------
# 10. Comparação com o prêmio vigente
# ------------------------------------------------------------

base["variacao_preco_abs"] = (
    base["premio_comercial_indicado"]
    -
    base["premio_vigente"]
)

base["variacao_preco_pct"] = (
    base["premio_comercial_indicado"]
    /
    base["premio_vigente"]
    - 1
)

base["indice_preco_indicado_vigente"] = (
    base["premio_comercial_indicado"]
    /
    base["premio_vigente"]
)

base["indice_vigente_indicado"] = (
    base["premio_vigente"]
    /
    base["premio_comercial_indicado"]
)

# Loss Ratio técnico sob o novo preço comercial
base["lr_tecnico_preco_indicado"] = (
    base["premio_puro_anual"]
    /
    base["premio_comercial_indicado"]
)


# ------------------------------------------------------------
# 11. Classificação da necessidade de reajuste
# ------------------------------------------------------------

def classificar_reajuste(x):
    if x <= -0.10:
        return "Redução >= 10%"
    elif x < -0.03:
        return "Redução de 3% a 10%"
    elif x <= 0.03:
        return "Próximo do vigente (+/-3%)"
    elif x < 0.10:
        return "Aumento de 3% a 10%"
    else:
        return "Aumento >= 10%"

base["faixa_reajuste"] = (
    base["variacao_preco_pct"]
    .apply(classificar_reajuste)
)

ordem_reajuste = [
    "Redução >= 10%",
    "Redução de 3% a 10%",
    "Próximo do vigente (+/-3%)",
    "Aumento de 3% a 10%",
    "Aumento >= 10%"
]

base["faixa_reajuste"] = pd.Categorical(
    base["faixa_reajuste"],
    categories=ordem_reajuste,
    ordered=True
)


# ------------------------------------------------------------
# 12. Prêmios correspondentes à exposição observada
# ------------------------------------------------------------

base["premio_comercial_indicado_periodo"] = (
    base["premio_comercial_indicado"]
    *
    base["exposicao"]
)

base["margem_alvo_periodo"] = (
    base["margem_alvo_valor"]
    *
    base["exposicao"]
)

base["despesa_fixa_periodo"] = (
    base["despesa_fixa_valor"]
    *
    base["exposicao"]
)

base["comissao_periodo"] = (
    base["comissao_valor"]
    *
    base["exposicao"]
)

base["tributos_encargos_periodo"] = (
    base["tributos_encargos_valor"]
    *
    base["exposicao"]
)

base["despesa_variavel_periodo"] = (
    base["despesa_variavel_valor"]
    *
    base["exposicao"]
)


# ------------------------------------------------------------
# 13. Resumo executivo da carteira
# ------------------------------------------------------------

premio_puro_total_periodo = base["custo_esperado_periodo"].sum()
premio_comercial_total_periodo = (
    base["premio_comercial_indicado_periodo"].sum()
)
premio_ganho_total = base["premio_ganho"].sum()

premio_puro_medio = base["premio_puro_anual"].mean()
premio_comercial_medio = base["premio_comercial_indicado"].mean()
premio_vigente_medio = base["premio_vigente"].mean()

variacao_media_pct = (
    premio_comercial_medio
    /
    premio_vigente_medio
    - 1
)

lr_tecnico_novo_agregado = (
    premio_puro_total_periodo
    /
    premio_comercial_total_periodo
)

margem_alvo_total_periodo = base["margem_alvo_periodo"].sum()
comissao_total_periodo = base["comissao_periodo"].sum()
tributos_total_periodo = base["tributos_encargos_periodo"].sum()
despesa_variavel_total_periodo = base["despesa_variavel_periodo"].sum()
despesa_fixa_total_periodo = base["despesa_fixa_periodo"].sum()

print("\n================ RESUMO EXECUTIVO — ETAPA 7 ================")
print(f"Prêmio puro anual médio: R$ {premio_puro_medio:,.2f}")
print(f"Prêmio comercial indicado médio: R$ {premio_comercial_medio:,.2f}")
print(f"Prêmio vigente médio: R$ {premio_vigente_medio:,.2f}")
print(f"Variação média indicada vs vigente: {variacao_media_pct:.2%}")
print(f"Prêmio comercial indicado no período: R$ {premio_comercial_total_periodo:,.2f}")
print(f"Prêmio ganho vigente no período: R$ {premio_ganho_total:,.2f}")
print(f"LR técnico agregado sob preço indicado: {lr_tecnico_novo_agregado:.2%}")
print(f"Comissão no período: R$ {comissao_total_periodo:,.2f}")
print(f"Tributos/encargos no período: R$ {tributos_total_periodo:,.2f}")
print(f"Despesa variável no período: R$ {despesa_variavel_total_periodo:,.2f}")
print(f"Despesa fixa no período: R$ {despesa_fixa_total_periodo:,.2f}")
print(f"Margem-alvo no período: R$ {margem_alvo_total_periodo:,.2f}")


# ------------------------------------------------------------
# 14. Fechamento agregado da equação comercial
# ------------------------------------------------------------

fechamento_agregado = (
    premio_puro_total_periodo
    + comissao_total_periodo
    + tributos_total_periodo
    + despesa_variavel_total_periodo
    + despesa_fixa_total_periodo
    + margem_alvo_total_periodo
)

diferenca_fechamento_agregado = (
    premio_comercial_total_periodo
    -
    fechamento_agregado
)

print("\n================ FECHAMENTO DO PRÊMIO COMERCIAL ================")
print(
    f"Prêmio comercial indicado: "
    f"R$ {premio_comercial_total_periodo:,.2f}"
)
print(
    f"Soma prêmio puro + carregamentos: "
    f"R$ {fechamento_agregado:,.2f}"
)
print(
    f"Diferença de fechamento: "
    f"R$ {diferenca_fechamento_agregado:,.6f}"
)


# ------------------------------------------------------------
# 15. Distribuição do prêmio comercial indicado
# ------------------------------------------------------------

resumo_distribuicao = (
    base["premio_comercial_indicado"]
    .describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n================ DISTRIBUIÇÃO DO PRÊMIO COMERCIAL ================")
print(resumo_distribuicao)


# ------------------------------------------------------------
# 16. Distribuição da necessidade de reajuste
# ------------------------------------------------------------

resumo_reajuste = (
    base.groupby("faixa_reajuste", observed=False)
    .agg(
        apolices=("id_apolice", "count"),
        premio_vigente_medio=("premio_vigente", "mean"),
        premio_indicado_medio=("premio_comercial_indicado", "mean"),
        variacao_media_pct=("variacao_preco_pct", "mean"),
        premio_puro_medio=("premio_puro_anual", "mean")
    )
)

resumo_reajuste["percentual_carteira"] = (
    resumo_reajuste["apolices"]
    /
    len(base)
)

print("\n================ FAIXAS DE REAJUSTE ================")
print(resumo_reajuste)


# ------------------------------------------------------------
# 17. Faixa etária para análise
# ------------------------------------------------------------

base["faixa_etaria_analise"] = pd.cut(
    base["idade_condutor"],
    bins=[17, 24, 34, 44, 54, 64, 80],
    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    include_lowest=True
)


# ------------------------------------------------------------
# 18. Função de resumo por segmento
# ------------------------------------------------------------

def resumo_comercial_por_segmento(df, coluna):
    resultado = (
        df.groupby(coluna, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            exposicao=("exposicao", "sum"),
            premio_puro_medio=("premio_puro_anual", "mean"),
            premio_vigente_medio=("premio_vigente", "mean"),
            premio_indicado_medio=("premio_comercial_indicado", "mean"),
            premio_ganho=("premio_ganho", "sum"),
            premio_indicado_periodo=("premio_comercial_indicado_periodo", "sum"),
            custo_esperado=("custo_esperado_periodo", "sum"),
            margem_alvo_periodo=("margem_alvo_periodo", "sum")
        )
    )

    resultado["variacao_indicada_pct"] = (
        resultado["premio_indicado_medio"]
        /
        resultado["premio_vigente_medio"]
        - 1
    )

    resultado["lr_tecnico_vigente"] = (
        resultado["custo_esperado"]
        /
        resultado["premio_ganho"].replace(0, np.nan)
    )

    resultado["lr_tecnico_indicado"] = (
        resultado["custo_esperado"]
        /
        resultado["premio_indicado_periodo"].replace(0, np.nan)
    )

    resultado["adequacao_vigente_vs_indicado"] = (
        resultado["premio_vigente_medio"]
        /
        resultado["premio_indicado_medio"].replace(0, np.nan)
    )

    return resultado


# ------------------------------------------------------------
# 19. Resumos segmentados
# ------------------------------------------------------------

resumo_regiao = resumo_comercial_por_segmento(base, "regiao")
resumo_tipo_uso = resumo_comercial_por_segmento(base, "tipo_uso")
resumo_tipo_veiculo = resumo_comercial_por_segmento(base, "tipo_veiculo")
resumo_cobertura = resumo_comercial_por_segmento(base, "nivel_cobertura")
resumo_bonus = resumo_comercial_por_segmento(base, "classe_bonus")
resumo_faixa_etaria = resumo_comercial_por_segmento(
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
    print(f"\n================ PRÊMIO COMERCIAL POR {nome.upper()} ================")
    print(tabela)

    tabela.to_csv(
        PASTA_TABELAS / f"premio_comercial_por_{nome}.csv",
        encoding="utf-8-sig"
    )


# ------------------------------------------------------------
# 20. Exportação dos principais resultados
# ------------------------------------------------------------

pd.DataFrame({
    "parametro": [
        "despesa_fixa_anual",
        "comissao_pct",
        "tributos_encargos_pct",
        "despesa_variavel_pct",
        "margem_alvo_pct",
        "soma_carregamentos_pct"
    ],
    "valor": [
        DESPESA_FIXA_ANUAL,
        COMISSAO_PCT,
        TRIBUTOS_ENCARGOS_PCT,
        DESPESA_VARIAVEL_PCT,
        MARGEM_ALVO_PCT,
        SOMA_CARREGAMENTOS_PCT
    ]
}).to_csv(
    PASTA_TABELAS / "parametros_premio_comercial.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "metrica": [
        "premio_puro_medio",
        "premio_comercial_indicado_medio",
        "premio_vigente_medio",
        "variacao_media_pct",
        "premio_puro_total_periodo",
        "premio_comercial_total_periodo",
        "premio_ganho_total",
        "lr_tecnico_novo_agregado",
        "comissao_total_periodo",
        "tributos_total_periodo",
        "despesa_variavel_total_periodo",
        "despesa_fixa_total_periodo",
        "margem_alvo_total_periodo",
        "diferenca_fechamento_agregado"
    ],
    "valor": [
        premio_puro_medio,
        premio_comercial_medio,
        premio_vigente_medio,
        variacao_media_pct,
        premio_puro_total_periodo,
        premio_comercial_total_periodo,
        premio_ganho_total,
        lr_tecnico_novo_agregado,
        comissao_total_periodo,
        tributos_total_periodo,
        despesa_variavel_total_periodo,
        despesa_fixa_total_periodo,
        margem_alvo_total_periodo,
        diferenca_fechamento_agregado
    ]
}).to_csv(
    PASTA_TABELAS / "resumo_premio_comercial.csv",
    index=False,
    encoding="utf-8-sig"
)

resumo_distribuicao.to_csv(
    PASTA_TABELAS / "distribuicao_premio_comercial.csv",
    encoding="utf-8-sig"
)

resumo_reajuste.to_csv(
    PASTA_TABELAS / "faixas_reajuste_premio_comercial.csv",
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 21. Exportação da base completa
# ------------------------------------------------------------

colunas_exportacao = [
    "id_apolice",
    "exposicao",
    "idade_condutor",
    "faixa_idade_modelo",
    "faixa_etaria_analise",
    "idade_veiculo",
    "regiao",
    "tipo_veiculo",
    "tipo_uso",
    "nivel_cobertura",
    "classe_bonus",
    "premio_vigente",
    "premio_ganho",
    "quantidade_sinistros",
    "custo_total_sinistros",
    "frequencia_prevista",
    "severidade_prevista",
    "premio_puro_anual",
    "custo_esperado_periodo",
    "premio_comercial_indicado",
    "premio_comercial_indicado_periodo",
    "despesa_fixa_valor",
    "comissao_valor",
    "tributos_encargos_valor",
    "despesa_variavel_valor",
    "margem_alvo_valor",
    "total_carregamentos_valor",
    "variacao_preco_abs",
    "variacao_preco_pct",
    "indice_preco_indicado_vigente",
    "indice_vigente_indicado",
    "lr_tecnico_preco_indicado",
    "faixa_reajuste"
]

base[colunas_exportacao].to_csv(
    PASTA_TABELAS / "base_pricing_premio_comercial.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 22. Gráfico — distribuição do prêmio comercial
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.hist(base["premio_comercial_indicado"], bins=50)
plt.xlabel("Prêmio comercial indicado anual (R$)")
plt.ylabel("Quantidade de apólices")
plt.title("Distribuição do prêmio comercial indicado")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "distribuicao_premio_comercial.png",
    dpi=150
)
plt.show()


# ------------------------------------------------------------
# 23. Gráfico — vigente × indicado
# ------------------------------------------------------------

plt.figure(figsize=(8, 8))
plt.scatter(
    base["premio_vigente"],
    base["premio_comercial_indicado"],
    alpha=0.20,
    s=10
)

limite = max(
    base["premio_vigente"].max(),
    base["premio_comercial_indicado"].max()
)

plt.plot(
    [0, limite],
    [0, limite],
    linestyle="--"
)

plt.xlabel("Prêmio vigente anual (R$)")
plt.ylabel("Prêmio comercial indicado anual (R$)")
plt.title("Prêmio vigente × prêmio comercial indicado")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "premio_vigente_vs_comercial_indicado.png",
    dpi=150
)
plt.show()


# ------------------------------------------------------------
# 24. Gráfico — composição do prêmio comercial médio
# ------------------------------------------------------------

componentes_medios = pd.Series({
    "Prêmio puro": base["premio_puro_anual"].mean(),
    "Despesa fixa": base["despesa_fixa_valor"].mean(),
    "Comissão": base["comissao_valor"].mean(),
    "Tributos/encargos": base["tributos_encargos_valor"].mean(),
    "Despesa variável": base["despesa_variavel_valor"].mean(),
    "Margem-alvo": base["margem_alvo_valor"].mean()
})

plt.figure(figsize=(10, 5))
plt.bar(
    componentes_medios.index,
    componentes_medios.values
)
plt.ylabel("Valor médio anual por apólice (R$)")
plt.title("Composição média do prêmio comercial indicado")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "composicao_premio_comercial_medio.png",
    dpi=150
)
plt.show()


# ------------------------------------------------------------
# 25. Gráfico — necessidade de reajuste
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(
    resumo_reajuste.index.astype(str),
    resumo_reajuste["percentual_carteira"]
)
plt.ylabel("Percentual da carteira")
plt.xlabel("Faixa de reajuste")
plt.title("Distribuição da necessidade de reajuste tarifário")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "faixas_reajuste_premio_comercial.png",
    dpi=150
)
plt.show()


# ------------------------------------------------------------
# 26. Gráfico — preço vigente × indicado por tipo de veículo
# ------------------------------------------------------------

categorias = resumo_tipo_veiculo.index.astype(str)
x = np.arange(len(categorias))
largura = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - largura / 2,
    resumo_tipo_veiculo["premio_vigente_medio"],
    width=largura,
    label="Vigente"
)

plt.bar(
    x + largura / 2,
    resumo_tipo_veiculo["premio_indicado_medio"],
    width=largura,
    label="Indicado"
)

plt.xticks(x, categorias, rotation=45)
plt.ylabel("Prêmio anual médio (R$)")
plt.title("Prêmio vigente × indicado por tipo de veículo")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "premio_vigente_vs_indicado_tipo_veiculo.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 27. Encerramento
# ------------------------------------------------------------

print("\nEtapa 7 concluída com sucesso.")