from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# ETAPA 12 — RETORNO SOBRE CAPITAL E DECISÃO FINAL
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
#
# OBJETIVO
# Consolidar os resultados das etapas anteriores em uma estrutura
# gerencial de decisão entre as estratégias:
#
#   Growth
#   Margin
#   Capital-efficient
#
# A etapa combina:
#
#   - retenção / crescimento;
#   - resultado econômico esperado;
#   - capital econômico VaR 99,5%;
#   - capital econômico TVaR 99,5%;
#   - retorno sobre capital;
#   - risco de cauda;
#   - Combined Ratio;
#   - criação de valor econômico;
#   - restrições de apetite a risco;
#   - score multicritério;
#   - análise de Pareto.
#
# IMPORTANTE
# A decisão final NÃO é tratada como "verdade matemática".
# Ela depende:
#
#   1. do objetivo estratégico;
#   2. do custo de capital;
#   3. do apetite a risco;
#   4. dos pesos atribuídos a crescimento, margem e capital.
#
# Portanto, o script separa:
#
#   RESULTADO TÉCNICO
#   de
#   PREFERÊNCIA GERENCIAL.
#
# Todos os dados e parâmetros são sintéticos/didáticos.
# ============================================================


# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

PASTA_TABELAS = Path("../outputs/tabelas")
PASTA_FIGURAS = Path("../outputs/figuras")

ARQUIVO_ETAPA9 = (
    PASTA_TABELAS / "comparacao_estrategias_pricing.csv"
)

ARQUIVO_CAPITAL_995 = (
    PASTA_TABELAS / "capital_economico_995_roc.csv"
)

ARQUIVO_CAUDA = (
    PASTA_TABELAS / "resumo_cauda_estrategias.csv"
)

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Parâmetros gerenciais
# ------------------------------------------------------------
#
# Taxa mínima de retorno exigida sobre capital econômico.
# Valor didático/sintético.
#
# Criação de valor econômico:
#
# EVA = Resultado esperado - h * Capital
#
# onde h é o hurdle rate.

HURDLE_RATE = 0.20

# ------------------------------------------------------------
# 2.1 Apetite a risco — cenário-base
# ------------------------------------------------------------
#
# Uma estratégia é considerada aderente ao apetite se:
#
#   retenção >= 60%
#   Combined Ratio P99,5 <= 100%
#   probabilidade de resultado negativo <= 5%
#
# Estes limites são didáticos e parametrizáveis.

RETENCAO_MINIMA = 0.60
COMBINED_RATIO_P995_MAX = 1.00
PROB_RESULTADO_NEGATIVO_MAX = 0.05


# ------------------------------------------------------------
# 2.2 Pesos do score multicritério
# ------------------------------------------------------------
#
# O score é apenas uma ferramenta de decisão.
#
# Quanto maior:
#   - resultado;
#   - retenção;
#   - RoC;
#   - EVA;
#
# melhor.
#
# Quanto menor:
#   - capital;
#   - Combined Ratio P99,5;
#
# melhor.

PESOS = {
    "resultado": 0.25,
    "retencao": 0.15,
    "roc": 0.25,
    "eva": 0.20,
    "capital": 0.10,
    "cauda": 0.05
}

if not np.isclose(sum(PESOS.values()), 1.0):
    raise ValueError(
        "Os pesos do score multicritério devem somar 1."
    )


# ------------------------------------------------------------
# 3. Leitura dos resultados das etapas anteriores
# ------------------------------------------------------------

for arquivo in [
    ARQUIVO_ETAPA9,
    ARQUIVO_CAPITAL_995,
    ARQUIVO_CAUDA
]:
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Arquivo necessário não encontrado: {arquivo}"
        )

estrategias_etapa9 = pd.read_csv(
    ARQUIVO_ETAPA9
)

capital_995 = pd.read_csv(
    ARQUIVO_CAPITAL_995
)

resumo_cauda = pd.read_csv(
    ARQUIVO_CAUDA
)

print("\n================ ARQUIVOS LIDOS ================")
print(f"Etapa 9: {estrategias_etapa9.shape}")
print(f"Capital 99,5%: {capital_995.shape}")
print(f"Resumo de cauda: {resumo_cauda.shape}")


# ------------------------------------------------------------
# 4. Validação das colunas necessárias
# ------------------------------------------------------------

colunas_etapa9 = [
    "estrategia",
    "retencao_media",
    "apolices_esperadas_retidas"
]

colunas_capital = [
    "estrategia",
    "resultado_medio",
    "capital_economico_VaR",
    "capital_economico_TVaR",
    "capital_resultado",
    "RoC_VaR_995",
    "RoC_TVaR_995"
]

colunas_cauda = [
    "estrategia",
    "prob_resultado_negativo",
    "prob_combined_ratio_acima_100",
    "combined_ratio_p95",
    "combined_ratio_p99",
    "combined_ratio_p995"
]

for coluna in colunas_etapa9:
    if coluna not in estrategias_etapa9.columns:
        raise KeyError(
            f"Coluna ausente no arquivo da Etapa 9: {coluna}"
        )

for coluna in colunas_capital:
    if coluna not in capital_995.columns:
        raise KeyError(
            f"Coluna ausente no arquivo de capital: {coluna}"
        )

for coluna in colunas_cauda:
    if coluna not in resumo_cauda.columns:
        raise KeyError(
            f"Coluna ausente no resumo de cauda: {coluna}"
        )


# ------------------------------------------------------------
# 5. Consolidação
# ------------------------------------------------------------

base_decisao = (
    estrategias_etapa9[colunas_etapa9]
    .merge(
        capital_995[colunas_capital],
        on="estrategia",
        how="inner"
    )
    .merge(
        resumo_cauda[colunas_cauda],
        on="estrategia",
        how="inner"
    )
)

if len(base_decisao) != 3:
    raise ValueError(
        "Esperávamos exatamente três estratégias consolidadas."
    )

print("\n================ BASE CONSOLIDADA ================")
print(base_decisao)


# ------------------------------------------------------------
# 6. Criação de valor econômico — EVA
# ------------------------------------------------------------
#
# EVA_VaR =
# Resultado médio - hurdle * Capital VaR 99,5%
#
# EVA_TVaR =
# Resultado médio - hurdle * Capital TVaR 99,5%

base_decisao["custo_capital_VaR"] = (
    HURDLE_RATE
    *
    base_decisao["capital_economico_VaR"]
)

base_decisao["custo_capital_TVaR"] = (
    HURDLE_RATE
    *
    base_decisao["capital_economico_TVaR"]
)

base_decisao["EVA_VaR"] = (
    base_decisao["resultado_medio"]
    -
    base_decisao["custo_capital_VaR"]
)

base_decisao["EVA_TVaR"] = (
    base_decisao["resultado_medio"]
    -
    base_decisao["custo_capital_TVaR"]
)


# ------------------------------------------------------------
# 7. Margem de segurança de cauda
# ------------------------------------------------------------
#
# Se Combined Ratio P99,5 < 100%:
#
# margem de segurança positiva.
#
# Exemplo:
# CR P99,5 = 97%
# margem = 3 p.p.

base_decisao["margem_seguranca_p995"] = (
    1
    -
    base_decisao["combined_ratio_p995"]
)


# ------------------------------------------------------------
# 8. Testes de apetite a risco
# ------------------------------------------------------------

base_decisao["cumpre_retencao_minima"] = (
    base_decisao["retencao_media"]
    >=
    RETENCAO_MINIMA
)

base_decisao["cumpre_combined_ratio_cauda"] = (
    base_decisao["combined_ratio_p995"]
    <=
    COMBINED_RATIO_P995_MAX
)

base_decisao["cumpre_prob_resultado_negativo"] = (
    base_decisao["prob_resultado_negativo"]
    <=
    PROB_RESULTADO_NEGATIVO_MAX
)

base_decisao["aderente_apetite"] = (
    base_decisao[
        [
            "cumpre_retencao_minima",
            "cumpre_combined_ratio_cauda",
            "cumpre_prob_resultado_negativo"
        ]
    ]
    .all(axis=1)
)

print("\n================ APETITE A RISCO ================")
print(
    base_decisao[
        [
            "estrategia",
            "retencao_media",
            "combined_ratio_p995",
            "prob_resultado_negativo",
            "aderente_apetite"
        ]
    ]
)


# ------------------------------------------------------------
# 9. Funções de normalização multicritério
# ------------------------------------------------------------

def normalizar_beneficio(serie):
    """
    Quanto maior, melhor.
    Normalização min-max entre 0 e 1.
    """
    minimo = serie.min()
    maximo = serie.max()

    if np.isclose(maximo, minimo):
        return pd.Series(
            np.ones(len(serie)),
            index=serie.index
        )

    return (
        (serie - minimo)
        /
        (maximo - minimo)
    )


def normalizar_custo(serie):
    """
    Quanto menor, melhor.
    """
    return 1 - normalizar_beneficio(serie)


# ------------------------------------------------------------
# 10. Componentes normalizados do score
# ------------------------------------------------------------

base_decisao["score_resultado"] = normalizar_beneficio(
    base_decisao["resultado_medio"]
)

base_decisao["score_retencao"] = normalizar_beneficio(
    base_decisao["retencao_media"]
)

base_decisao["score_roc"] = normalizar_beneficio(
    base_decisao["RoC_VaR_995"]
)

base_decisao["score_eva"] = normalizar_beneficio(
    base_decisao["EVA_VaR"]
)

base_decisao["score_capital"] = normalizar_custo(
    base_decisao["capital_economico_VaR"]
)

base_decisao["score_cauda"] = normalizar_custo(
    base_decisao["combined_ratio_p995"]
)


# ------------------------------------------------------------
# 11. Score multicritério
# ------------------------------------------------------------

base_decisao["score_multicriterio"] = (
    PESOS["resultado"]
    *
    base_decisao["score_resultado"]

    +
    PESOS["retencao"]
    *
    base_decisao["score_retencao"]

    +
    PESOS["roc"]
    *
    base_decisao["score_roc"]

    +
    PESOS["eva"]
    *
    base_decisao["score_eva"]

    +
    PESOS["capital"]
    *
    base_decisao["score_capital"]

    +
    PESOS["cauda"]
    *
    base_decisao["score_cauda"]
)

base_decisao["rank_multicriterio"] = (
    base_decisao["score_multicriterio"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ------------------------------------------------------------
# 12. Rankings individuais
# ------------------------------------------------------------

base_decisao["rank_resultado"] = (
    base_decisao["resultado_medio"]
    .rank(
        ascending=False,
        method="min"
    )
)

base_decisao["rank_retencao"] = (
    base_decisao["retencao_media"]
    .rank(
        ascending=False,
        method="min"
    )
)

base_decisao["rank_menor_capital"] = (
    base_decisao["capital_economico_VaR"]
    .rank(
        ascending=True,
        method="min"
    )
)

base_decisao["rank_roc"] = (
    base_decisao["RoC_VaR_995"]
    .rank(
        ascending=False,
        method="min"
    )
)

base_decisao["rank_eva"] = (
    base_decisao["EVA_VaR"]
    .rank(
        ascending=False,
        method="min"
    )
)

base_decisao["rank_cauda"] = (
    base_decisao["combined_ratio_p995"]
    .rank(
        ascending=True,
        method="min"
    )
)


# ------------------------------------------------------------
# 13. Análise de dominância de Pareto
# ------------------------------------------------------------
#
# Uma estratégia A domina B se for:
#
#   >= em resultado
#   >= em retenção
#   >= em RoC
#   <= em capital
#   <= em Combined Ratio P99,5
#
# e estritamente melhor em pelo menos um critério.

def domina(linha_a, linha_b):
    condicoes_nao_piores = [
        linha_a["resultado_medio"]
        >= linha_b["resultado_medio"],

        linha_a["retencao_media"]
        >= linha_b["retencao_media"],

        linha_a["RoC_VaR_995"]
        >= linha_b["RoC_VaR_995"],

        linha_a["capital_economico_VaR"]
        <= linha_b["capital_economico_VaR"],

        linha_a["combined_ratio_p995"]
        <= linha_b["combined_ratio_p995"]
    ]

    estritamente_melhor = [
        linha_a["resultado_medio"]
        > linha_b["resultado_medio"],

        linha_a["retencao_media"]
        > linha_b["retencao_media"],

        linha_a["RoC_VaR_995"]
        > linha_b["RoC_VaR_995"],

        linha_a["capital_economico_VaR"]
        < linha_b["capital_economico_VaR"],

        linha_a["combined_ratio_p995"]
        < linha_b["combined_ratio_p995"]
    ]

    return (
        all(condicoes_nao_piores)
        and any(estritamente_melhor)
    )


dominada_por = {}

for _, linha_b in base_decisao.iterrows():
    nome_b = linha_b["estrategia"]
    dominadores = []

    for _, linha_a in base_decisao.iterrows():
        nome_a = linha_a["estrategia"]

        if nome_a == nome_b:
            continue

        if domina(linha_a, linha_b):
            dominadores.append(nome_a)

    dominada_por[nome_b] = dominadores

base_decisao["dominada_pareto"] = (
    base_decisao["estrategia"]
    .map(
        lambda x: len(dominada_por[x]) > 0
    )
)

base_decisao["dominada_por"] = (
    base_decisao["estrategia"]
    .map(
        lambda x: ", ".join(dominada_por[x])
        if dominada_por[x]
        else ""
    )
)


# ------------------------------------------------------------
# 14. Fronteira eficiente simples
# ------------------------------------------------------------

fronteira_pareto = base_decisao[
    ~base_decisao["dominada_pareto"]
].copy()

print("\n================ FRONTEIRA DE PARETO ================")
print(
    fronteira_pareto[
        [
            "estrategia",
            "resultado_medio",
            "retencao_media",
            "capital_economico_VaR",
            "RoC_VaR_995",
            "combined_ratio_p995"
        ]
    ]
)


# ------------------------------------------------------------
# 15. Recomendação condicionada ao apetite
# ------------------------------------------------------------
#
# Regra:
# 1. prioriza estratégias aderentes ao apetite;
# 2. dentre elas, escolhe maior score multicritério.
#
# Se nenhuma estratégia for aderente:
# usa maior score geral, mas sinaliza exceção.

elegiveis = base_decisao[
    base_decisao["aderente_apetite"]
].copy()

if not elegiveis.empty:
    estrategia_recomendada = (
        elegiveis
        .sort_values(
            "score_multicriterio",
            ascending=False
        )
        .iloc[0]
    )

    status_recomendacao = (
        "Recomendação dentro do apetite a risco"
    )
else:
    estrategia_recomendada = (
        base_decisao
        .sort_values(
            "score_multicriterio",
            ascending=False
        )
        .iloc[0]
    )

    status_recomendacao = (
        "Nenhuma estratégia cumpre integralmente "
        "o apetite; recomendação baseada apenas no score"
    )


# ------------------------------------------------------------
# 16. Resumo executivo
# ------------------------------------------------------------

print("\n================ SCORE MULTICRITÉRIO ================")
print(
    base_decisao[
        [
            "estrategia",
            "score_multicriterio",
            "rank_multicriterio",
            "aderente_apetite"
        ]
    ]
    .sort_values(
        "rank_multicriterio"
    )
)

print("\n================ RECOMENDAÇÃO DO CENÁRIO-BASE ================")
print(status_recomendacao)
print(
    f"Estratégia: "
    f"{estrategia_recomendada['estrategia']}"
)
print(
    f"Score multicritério: "
    f"{estrategia_recomendada['score_multicriterio']:.4f}"
)
print(
    f"Retenção: "
    f"{estrategia_recomendada['retencao_media']:.2%}"
)
print(
    f"Resultado médio: "
    f"R$ {estrategia_recomendada['resultado_medio']:,.2f}"
)
print(
    f"Capital econômico VaR 99,5%: "
    f"R$ {estrategia_recomendada['capital_economico_VaR']:,.2f}"
)
print(
    f"RoC VaR 99,5%: "
    f"{estrategia_recomendada['RoC_VaR_995']:.4f}"
)
print(
    f"EVA VaR: "
    f"R$ {estrategia_recomendada['EVA_VaR']:,.2f}"
)
print(
    f"Combined Ratio P99,5: "
    f"{estrategia_recomendada['combined_ratio_p995']:.2%}"
)


# ------------------------------------------------------------
# 17. Matriz gerencial resumida
# ------------------------------------------------------------

matriz_gerencial = base_decisao[
    [
        "estrategia",
        "retencao_media",
        "apolices_esperadas_retidas",
        "resultado_medio",
        "capital_economico_VaR",
        "capital_economico_TVaR",
        "RoC_VaR_995",
        "RoC_TVaR_995",
        "EVA_VaR",
        "EVA_TVaR",
        "prob_resultado_negativo",
        "combined_ratio_p995",
        "margem_seguranca_p995",
        "aderente_apetite",
        "score_multicriterio",
        "rank_multicriterio",
        "dominada_pareto",
        "dominada_por"
    ]
].copy()

matriz_gerencial = matriz_gerencial.sort_values(
    "rank_multicriterio"
)

print("\n================ MATRIZ GERENCIAL ================")
print(matriz_gerencial)


# ------------------------------------------------------------
# 18. Tabela de premissas de decisão
# ------------------------------------------------------------

premissas_decisao = pd.DataFrame({
    "parametro": [
        "hurdle_rate",
        "retencao_minima",
        "combined_ratio_p995_max",
        "prob_resultado_negativo_max",
        "peso_resultado",
        "peso_retencao",
        "peso_roc",
        "peso_eva",
        "peso_capital",
        "peso_cauda"
    ],
    "valor": [
        HURDLE_RATE,
        RETENCAO_MINIMA,
        COMBINED_RATIO_P995_MAX,
        PROB_RESULTADO_NEGATIVO_MAX,
        PESOS["resultado"],
        PESOS["retencao"],
        PESOS["roc"],
        PESOS["eva"],
        PESOS["capital"],
        PESOS["cauda"]
    ]
})


# ------------------------------------------------------------
# 19. Exportações
# ------------------------------------------------------------

base_decisao.to_csv(
    PASTA_TABELAS / "decisao_final_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

matriz_gerencial.to_csv(
    PASTA_TABELAS / "matriz_gerencial_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

fronteira_pareto.to_csv(
    PASTA_TABELAS / "fronteira_pareto_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

premissas_decisao.to_csv(
    PASTA_TABELAS / "premissas_decisao_etapa12.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "campo": [
        "status",
        "estrategia_recomendada",
        "score_multicriterio",
        "aderente_apetite",
        "resultado_medio",
        "capital_economico_VaR_995",
        "capital_economico_TVaR_995",
        "RoC_VaR_995",
        "RoC_TVaR_995",
        "EVA_VaR",
        "EVA_TVaR",
        "retencao_media",
        "combined_ratio_p995"
    ],
    "valor": [
        status_recomendacao,
        estrategia_recomendada["estrategia"],
        estrategia_recomendada["score_multicriterio"],
        estrategia_recomendada["aderente_apetite"],
        estrategia_recomendada["resultado_medio"],
        estrategia_recomendada["capital_economico_VaR"],
        estrategia_recomendada["capital_economico_TVaR"],
        estrategia_recomendada["RoC_VaR_995"],
        estrategia_recomendada["RoC_TVaR_995"],
        estrategia_recomendada["EVA_VaR"],
        estrategia_recomendada["EVA_TVaR"],
        estrategia_recomendada["retencao_media"],
        estrategia_recomendada["combined_ratio_p995"]
    ]
}).to_csv(
    PASTA_TABELAS / "recomendacao_estrategia_final.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 20. Gráfico — Resultado × Capital Econômico
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    base_decisao["capital_economico_VaR"],
    base_decisao["resultado_medio"],
    s=120
)

for _, linha in base_decisao.iterrows():
    plt.annotate(
        linha["estrategia"],
        (
            linha["capital_economico_VaR"],
            linha["resultado_medio"]
        )
    )

plt.xlabel("Capital econômico VaR 99,5% (R$)")
plt.ylabel("Resultado médio (R$)")
plt.title("Resultado × Capital Econômico")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_resultado_vs_capital.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 21. Gráfico — Retenção × RoC
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    base_decisao["retencao_media"],
    base_decisao["RoC_VaR_995"],
    s=120
)

for _, linha in base_decisao.iterrows():
    plt.annotate(
        linha["estrategia"],
        (
            linha["retencao_media"],
            linha["RoC_VaR_995"]
        )
    )

plt.xlabel("Retenção média")
plt.ylabel("RoC VaR 99,5%")
plt.title("Trade-off: retenção × retorno sobre capital")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_retencao_vs_roc.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 22. Gráfico — EVA por estratégia
# ------------------------------------------------------------

ordem = [
    "Growth",
    "Capital-efficient",
    "Margin"
]

plot_base = (
    base_decisao
    .set_index("estrategia")
    .loc[ordem]
)

x = np.arange(
    len(ordem)
)

largura = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - largura / 2,
    plot_base["EVA_VaR"],
    width=largura,
    label="EVA — VaR"
)

plt.bar(
    x + largura / 2,
    plot_base["EVA_TVaR"],
    width=largura,
    label="EVA — TVaR"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xticks(
    x,
    ordem,
    rotation=15
)

plt.ylabel("Valor econômico agregado (R$)")
plt.title(
    f"EVA por estratégia — hurdle rate {HURDLE_RATE:.0%}"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_eva_estrategias.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 23. Gráfico — Score multicritério
# ------------------------------------------------------------

score_plot = (
    base_decisao
    .sort_values(
        "score_multicriterio",
        ascending=False
    )
)

plt.figure(figsize=(9, 5))

plt.bar(
    score_plot["estrategia"],
    score_plot["score_multicriterio"]
)

plt.ylabel("Score multicritério")
plt.title("Score estratégico consolidado")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_score_multicriterio.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 24. Gráfico — Combined Ratio P99,5%
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    plot_base.index,
    plot_base["combined_ratio_p995"]
)

plt.axhline(
    COMBINED_RATIO_P995_MAX,
    linestyle="--",
    label="Limite de apetite"
)

plt.ylabel("Combined Ratio P99,5%")
plt.title("Risco de cauda por estratégia")
plt.legend()
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_combined_ratio_p995.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 25. Gráfico — Radar normalizado em coordenadas polares
# ------------------------------------------------------------
#
# Dimensões:
# resultado, retenção, RoC, EVA, capital e cauda.
#
# Todas as dimensões são apresentadas já convertidas para
# "quanto maior, melhor".

dimensoes = [
    "score_resultado",
    "score_retencao",
    "score_roc",
    "score_eva",
    "score_capital",
    "score_cauda"
]

rotulos = [
    "Resultado",
    "Retenção",
    "RoC",
    "EVA",
    "Eficiência de capital",
    "Proteção de cauda"
]

n_dim = len(dimensoes)

angulos = np.linspace(
    0,
    2 * np.pi,
    n_dim,
    endpoint=False
).tolist()

angulos += angulos[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(
    111,
    polar=True
)

for _, linha in base_decisao.iterrows():
    valores = [
        linha[coluna]
        for coluna in dimensoes
    ]

    valores += valores[:1]

    ax.plot(
        angulos,
        valores,
        linewidth=2,
        label=linha["estrategia"]
    )

ax.set_xticks(
    angulos[:-1]
)

ax.set_xticklabels(
    rotulos
)

ax.set_ylim(
    0,
    1
)

plt.title("Perfil estratégico normalizado")
plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.30, 1.10)
)
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "etapa12_radar_estrategias.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 26. Encerramento
# ------------------------------------------------------------

print("\nEtapa 12 concluída com sucesso.")
print(
    "\nA recomendação depende explicitamente dos parâmetros "
    "de apetite, hurdle rate e pesos gerenciais definidos "
    "no início do script."
)