import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

PASTA_DADOS = Path("../data")
PASTA_TABELAS = Path("../outputs/tabelas")
PASTA_FIGURAS = Path("../outputs/figuras")
ARQUIVO_BANCO = PASTA_DADOS / "pricing.db"

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

conexao = sqlite3.connect(ARQUIVO_BANCO)
base_severidade = pd.read_sql_query(
    """
    SELECT
        s.id_sinistro,
        s.id_apolice,
        s.valor_sinistro,
        a.idade_condutor,
        a.idade_veiculo,
        a.regiao,
        a.tipo_veiculo,
        a.tipo_uso,
        a.nivel_cobertura,
        a.classe_bonus
    FROM sinistros s
    INNER JOIN apolices a
        ON s.id_apolice = a.id_apolice
    ORDER BY s.id_sinistro;
    """,
    conexao
)
conexao.close()

print("\nDimensão da base de severidade:")
print(base_severidade.shape)
print("\nPrimeiras linhas:")
print(base_severidade.head())

if base_severidade.empty:
    raise ValueError("A base de severidade está vazia.")
if base_severidade["valor_sinistro"].isna().any():
    raise ValueError("Existem valores de sinistro nulos.")
if (base_severidade["valor_sinistro"] <= 0).any():
    raise ValueError("Existem valores de sinistro menores ou iguais a zero.")
if base_severidade["id_sinistro"].duplicated().any():
    raise ValueError("Existem IDs de sinistro duplicados.")

print("\nQuantidade de sinistros:")
print(len(base_severidade))

print("\nResumo da severidade observada:")
print(
    base_severidade["valor_sinistro"].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

base_severidade["tipo_veiculo"] = pd.Categorical(
    base_severidade["tipo_veiculo"],
    categories=["Hatch", "Sedan", "SUV", "Picape", "Utilitario"]
)

base_severidade["nivel_cobertura"] = pd.Categorical(
    base_severidade["nivel_cobertura"],
    categories=["Intermediaria", "Basica", "Completa"]
)

formula_severidade = """
valor_sinistro
~
C(tipo_veiculo, Treatment(reference="Hatch"))
+
C(nivel_cobertura, Treatment(reference="Intermediaria"))
+
idade_veiculo
"""

modelo_nulo_gamma = smf.glm(
    formula="valor_sinistro ~ 1",
    data=base_severidade,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

modelo_gamma = smf.glm(
    formula=formula_severidade,
    data=base_severidade,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

print("\n================ RESUMO DO GLM GAMMA ================")
print(modelo_gamma.summary())

intervalos = modelo_gamma.conf_int()

tabela_coeficientes = pd.DataFrame({
    "coeficiente": modelo_gamma.params,
    "erro_padrao": modelo_gamma.bse,
    "p_valor": modelo_gamma.pvalues,
    "relatividade": np.exp(modelo_gamma.params),
    "relatividade_limite_inferior": np.exp(intervalos[0]),
    "relatividade_limite_superior": np.exp(intervalos[1])
})

print("\n================ COEFICIENTES E RELATIVIDADES ================")
print(tabela_coeficientes)

tabela_coeficientes.to_csv(
    PASTA_TABELAS / "coeficientes_glm_severidade.csv",
    encoding="utf-8-sig"
)

coeficientes_verdadeiros = {
    "Intercept": np.log(7500),
    'C(tipo_veiculo, Treatment(reference="Hatch"))[T.Sedan]': 0.08,
    'C(tipo_veiculo, Treatment(reference="Hatch"))[T.SUV]': 0.22,
    'C(tipo_veiculo, Treatment(reference="Hatch"))[T.Picape]': 0.30,
    'C(tipo_veiculo, Treatment(reference="Hatch"))[T.Utilitario]': 0.18,
    'C(nivel_cobertura, Treatment(reference="Intermediaria"))[T.Basica]': -0.10,
    'C(nivel_cobertura, Treatment(reference="Intermediaria"))[T.Completa]': 0.18,
    "idade_veiculo": -0.008
}

linhas = []
for termo, beta_verdadeiro in coeficientes_verdadeiros.items():
    beta_estimado = modelo_gamma.params.get(termo, np.nan)
    linhas.append({
        "termo": termo,
        "beta_verdadeiro": beta_verdadeiro,
        "beta_estimado": beta_estimado,
        "erro_estimacao": beta_estimado - beta_verdadeiro if pd.notna(beta_estimado) else np.nan,
        "relatividade_verdadeira": np.exp(beta_verdadeiro),
        "relatividade_estimada": np.exp(beta_estimado) if pd.notna(beta_estimado) else np.nan
    })

comparacao_coeficientes = pd.DataFrame(linhas)

print("\n================ VERDADEIRO × ESTIMADO ================")
print(comparacao_coeficientes)

comparacao_coeficientes.to_csv(
    PASTA_TABELAS / "comparacao_coeficientes_severidade.csv",
    index=False,
    encoding="utf-8-sig"
)

dispersao_pearson = modelo_gamma.pearson_chi2 / modelo_gamma.df_resid
dispersao_deviance = modelo_gamma.deviance / modelo_gamma.df_resid
phi_verdadeiro = 1 / 2.2

print("\n================ DIAGNÓSTICOS DO MODELO ================")
print(f"Scale estimada pelo modelo: {modelo_gamma.scale:.4f}")
print(f"Pearson Chi² / graus de liberdade: {dispersao_pearson:.4f}")
print(f"Deviance / graus de liberdade: {dispersao_deviance:.4f}")
print(f"Dispersão teórica aproximada do gerador: {phi_verdadeiro:.4f}")

reducao_deviance = modelo_nulo_gamma.deviance - modelo_gamma.deviance

print("\n================ MODELO NULO × MODELO COMPLETO ================")
print(f"AIC modelo nulo: {modelo_nulo_gamma.aic:.4f}")
print(f"AIC modelo completo: {modelo_gamma.aic:.4f}")
print(f"Deviance modelo nulo: {modelo_nulo_gamma.deviance:.4f}")
print(f"Deviance modelo completo: {modelo_gamma.deviance:.4f}")
print(f"Redução de Deviance: {reducao_deviance:.4f}")

pd.DataFrame({
    "modelo": ["Gamma nulo", "Gamma completo"],
    "AIC": [modelo_nulo_gamma.aic, modelo_gamma.aic],
    "deviance": [modelo_nulo_gamma.deviance, modelo_gamma.deviance],
    "pearson_chi2": [modelo_nulo_gamma.pearson_chi2, modelo_gamma.pearson_chi2],
    "df_resid": [modelo_nulo_gamma.df_resid, modelo_gamma.df_resid],
    "scale": [modelo_nulo_gamma.scale, modelo_gamma.scale]
}).to_csv(
    PASTA_TABELAS / "comparacao_modelos_severidade.csv",
    index=False,
    encoding="utf-8-sig"
)

base_severidade["severidade_prevista"] = modelo_gamma.predict(base_severidade)

sev_obs_total = base_severidade["valor_sinistro"].mean()
sev_prev_total = base_severidade["severidade_prevista"].mean()
oe_total = sev_obs_total / sev_prev_total

print("\n================ OBSERVADO × PREVISTO — TOTAL ================")
print(f"Severidade observada média: R$ {sev_obs_total:,.2f}")
print(f"Severidade prevista média: R$ {sev_prev_total:,.2f}")
print(f"Razão O/E: {oe_total:.4f}")

def validar_segmento_severidade(df, coluna):
    resultado = (
        df.groupby(coluna, observed=False)
        .agg(
            sinistros=("id_sinistro", "count"),
            severidade_observada=("valor_sinistro", "mean"),
            severidade_prevista=("severidade_prevista", "mean")
        )
    )
    resultado["OE"] = (
        resultado["severidade_observada"] /
        resultado["severidade_prevista"]
    )
    resultado["erro_relativo_previsao"] = (
        resultado["severidade_prevista"] -
        resultado["severidade_observada"]
    ) / resultado["severidade_observada"]

    # Tradução monetária da calibração por segmento.
    resultado["custo_observado"] = (
        resultado["sinistros"] *
        resultado["severidade_observada"]
    )

    resultado["custo_previsto"] = (
        resultado["sinistros"] *
        resultado["severidade_prevista"]
    )

    resultado["OE_custo"] = (
        resultado["custo_observado"] /
        resultado["custo_previsto"].replace(0, np.nan)
    )

    return resultado

validacao_veiculo = validar_segmento_severidade(base_severidade, "tipo_veiculo")
validacao_cobertura = validar_segmento_severidade(base_severidade, "nivel_cobertura")

base_severidade["faixa_idade_veiculo"] = pd.cut(
    base_severidade["idade_veiculo"],
    bins=[-1, 2, 5, 10, 15, np.inf],
    labels=["0-2", "3-5", "6-10", "11-15", "16+"]
)

validacao_idade_veiculo = validar_segmento_severidade(
    base_severidade,
    "faixa_idade_veiculo"
)

print("\n================ VALIDAÇÃO — TIPO DE VEÍCULO ================")
print(validacao_veiculo)
print("\n================ VALIDAÇÃO — COBERTURA ================")
print(validacao_cobertura)
print("\n================ VALIDAÇÃO — IDADE DO VEÍCULO ================")
print(validacao_idade_veiculo)

validacao_veiculo.to_csv(
    PASTA_TABELAS / "validacao_severidade_tipo_veiculo.csv",
    encoding="utf-8-sig"
)
validacao_cobertura.to_csv(
    PASTA_TABELAS / "validacao_severidade_cobertura.csv",
    encoding="utf-8-sig"
)
validacao_idade_veiculo.to_csv(
    PASTA_TABELAS / "validacao_severidade_idade_veiculo.csv",
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 17. Separação treino/teste por APÓLICE
# ------------------------------------------------------------
#
# A divisão é feita no nível da apólice, e não no nível do sinistro.
# Assim, todos os sinistros de uma mesma apólice ficam integralmente
# no treino OU no teste, evitando que características idênticas da
# mesma apólice apareçam nos dois conjuntos.
#
# Isso torna a validação metodologicamente mais rigorosa.

ids_apolices = (
    base_severidade["id_apolice"]
    .drop_duplicates()
)

ids_treino, ids_teste = train_test_split(
    ids_apolices,
    test_size=0.20,
    random_state=20260906
)

treino = base_severidade[
    base_severidade["id_apolice"].isin(ids_treino)
].copy()

teste = base_severidade[
    base_severidade["id_apolice"].isin(ids_teste)
].copy()

# Controle explícito contra sobreposição de apólices.
intersecao_ids = set(treino["id_apolice"]).intersection(
    set(teste["id_apolice"])
)

assert len(intersecao_ids) == 0, (
    "Há apólices presentes simultaneamente no treino e no teste."
)

print("\n================ DIVISÃO TREINO/TESTE POR APÓLICE ================")
print(f"Apólices únicas no treino: {treino['id_apolice'].nunique():,}")
print(f"Apólices únicas no teste: {teste['id_apolice'].nunique():,}")
print(f"Sinistros no treino: {len(treino):,}")
print(f"Sinistros no teste: {len(teste):,}")

modelo_gamma_treino = smf.glm(
    formula=formula_severidade,
    data=treino,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

teste["severidade_prevista"] = modelo_gamma_treino.predict(teste)

sev_obs_teste = teste["valor_sinistro"].mean()
sev_prev_teste = teste["severidade_prevista"].mean()
oe_teste = sev_obs_teste / sev_prev_teste

# Métricas monetárias agregadas.
# Como cada linha representa um sinistro, o custo previsto agregado
# é a soma das severidades médias previstas para os sinistros do teste.

custo_obs_teste = teste["valor_sinistro"].sum()
custo_prev_teste = teste["severidade_prevista"].sum()
oe_custo_teste = custo_obs_teste / custo_prev_teste

erro_monetario_teste = custo_prev_teste - custo_obs_teste
erro_monetario_pct_teste = (
    erro_monetario_teste / custo_obs_teste
)

mae_teste = mean_absolute_error(
    teste["valor_sinistro"],
    teste["severidade_prevista"]
)

rmse_teste = np.sqrt(
    mean_squared_error(
        teste["valor_sinistro"],
        teste["severidade_prevista"]
    )
)

print("\n================ VALIDAÇÃO FORA DA AMOSTRA ================")
print(f"Apólices únicas no treino: {treino['id_apolice'].nunique():,}")
print(f"Apólices únicas no teste: {teste['id_apolice'].nunique():,}")
print(f"Sinistros no treino: {len(treino):,}")
print(f"Sinistros no teste: {len(teste):,}")
print(f"Severidade média observada no teste: R$ {sev_obs_teste:,.2f}")
print(f"Severidade média prevista no teste: R$ {sev_prev_teste:,.2f}")
print(f"Razão O/E da severidade média: {oe_teste:.4f}")
print(f"Custo observado total no teste: R$ {custo_obs_teste:,.2f}")
print(f"Custo previsto total no teste: R$ {custo_prev_teste:,.2f}")
print(f"Razão O/E do custo total: {oe_custo_teste:.4f}")
print(f"Erro monetário previsto - observado: R$ {erro_monetario_teste:,.2f}")
print(f"Erro monetário relativo: {erro_monetario_pct_teste:.2%}")
print(f"MAE no teste: R$ {mae_teste:,.2f}")
print(f"RMSE no teste: R$ {rmse_teste:,.2f}")

pd.DataFrame({
    "metrica": [
        "apolices_unicas_treino",
        "apolices_unicas_teste",
        "sinistros_treino",
        "sinistros_teste",
        "severidade_observada_teste",
        "severidade_prevista_teste",
        "OE_severidade_media_teste",
        "custo_observado_total_teste",
        "custo_previsto_total_teste",
        "OE_custo_total_teste",
        "erro_monetario_teste",
        "erro_monetario_pct_teste",
        "MAE_teste",
        "RMSE_teste"
    ],
    "valor": [
        treino["id_apolice"].nunique(),
        teste["id_apolice"].nunique(),
        len(treino),
        len(teste),
        sev_obs_teste,
        sev_prev_teste,
        oe_teste,
        custo_obs_teste,
        custo_prev_teste,
        oe_custo_teste,
        erro_monetario_teste,
        erro_monetario_pct_teste,
        mae_teste,
        rmse_teste
    ]
}).to_csv(
    PASTA_TABELAS / "validacao_teste_glm_severidade.csv",
    index=False,
    encoding="utf-8-sig"
)

validacao_teste_veiculo = validar_segmento_severidade(teste, "tipo_veiculo")
validacao_teste_cobertura = validar_segmento_severidade(teste, "nivel_cobertura")

teste["faixa_idade_veiculo"] = pd.cut(
    teste["idade_veiculo"],
    bins=[-1, 2, 5, 10, 15, np.inf],
    labels=["0-2", "3-5", "6-10", "11-15", "16+"]
)

validacao_teste_idade_veiculo = validar_segmento_severidade(
    teste,
    "faixa_idade_veiculo"
)

print("\n================ TESTE — TIPO DE VEÍCULO ================")
print(validacao_teste_veiculo)
print("\n================ TESTE — COBERTURA ================")
print(validacao_teste_cobertura)
print("\n================ TESTE — IDADE DO VEÍCULO ================")
print(validacao_teste_idade_veiculo)

validacao_teste_veiculo.to_csv(
    PASTA_TABELAS / "validacao_teste_severidade_tipo_veiculo.csv",
    encoding="utf-8-sig"
)
validacao_teste_cobertura.to_csv(
    PASTA_TABELAS / "validacao_teste_severidade_cobertura.csv",
    encoding="utf-8-sig"
)
validacao_teste_idade_veiculo.to_csv(
    PASTA_TABELAS / "validacao_teste_severidade_idade_veiculo.csv",
    encoding="utf-8-sig"
)

def consolidar_validacao(tabela, nome_segmento):
    temp = tabela.reset_index().copy()
    primeira_coluna = temp.columns[0]
    temp = temp.rename(columns={primeira_coluna: "categoria"})
    temp.insert(0, "segmento", nome_segmento)
    return temp[
        [
            "segmento",
            "categoria",
            "sinistros",
            "severidade_observada",
            "severidade_prevista",
            "OE",
            "erro_relativo_previsao",
            "custo_observado",
            "custo_previsto",
            "OE_custo"
        ]
    ]

pd.concat(
    [
        consolidar_validacao(validacao_teste_veiculo, "tipo_veiculo"),
        consolidar_validacao(validacao_teste_cobertura, "nivel_cobertura"),
        consolidar_validacao(validacao_teste_idade_veiculo, "faixa_idade_veiculo")
    ],
    ignore_index=True
).to_csv(
    PASTA_TABELAS / "validacao_segmentada_teste_severidade.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 5))
plt.hist(base_severidade["valor_sinistro"], bins=50)
plt.xlabel("Valor do sinistro (R$)")
plt.ylabel("Frequência")
plt.title("Distribuição observada da severidade")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "distribuicao_severidade_etapa5.png", dpi=150)
plt.show()

tabela_plot = tabela_coeficientes.drop(index="Intercept", errors="ignore").copy()

plt.figure(figsize=(10, 6))
plt.barh(
    tabela_plot.index.astype(str),
    tabela_plot["relatividade"]
)
plt.axvline(1.0, linestyle="--")
plt.xlabel("Relatividade estimada")
plt.ylabel("Variável")
plt.title("Relatividades estimadas — GLM de severidade")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "relatividades_severidade.png", dpi=150)
plt.show()

def grafico_observado_previsto(tabela, titulo, nome_arquivo):
    categorias = tabela.index.astype(str)
    x = np.arange(len(categorias))
    largura = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - largura / 2,
        tabela["severidade_observada"],
        width=largura,
        label="Observada"
    )
    plt.bar(
        x + largura / 2,
        tabela["severidade_prevista"],
        width=largura,
        label="Prevista"
    )
    plt.xticks(x, categorias, rotation=45)
    plt.ylabel("Severidade média (R$)")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150)
    plt.show()

grafico_observado_previsto(
    validacao_veiculo,
    "Severidade observada × prevista por tipo de veículo",
    "observado_previsto_severidade_tipo_veiculo.png"
)
grafico_observado_previsto(
    validacao_cobertura,
    "Severidade observada × prevista por cobertura",
    "observado_previsto_severidade_cobertura.png"
)
grafico_observado_previsto(
    validacao_idade_veiculo,
    "Severidade observada × prevista por idade do veículo",
    "observado_previsto_severidade_idade_veiculo.png"
)
grafico_observado_previsto(
    validacao_teste_veiculo,
    "Teste — severidade observada × prevista por tipo de veículo",
    "teste_observado_previsto_severidade_tipo_veiculo.png"
)
grafico_observado_previsto(
    validacao_teste_cobertura,
    "Teste — severidade observada × prevista por cobertura",
    "teste_observado_previsto_severidade_cobertura.png"
)
grafico_observado_previsto(
    validacao_teste_idade_veiculo,
    "Teste — severidade observada × prevista por idade do veículo",
    "teste_observado_previsto_severidade_idade_veiculo.png"
)

base_severidade[
    [
        "id_sinistro",
        "id_apolice",
        "valor_sinistro",
        "tipo_veiculo",
        "nivel_cobertura",
        "idade_veiculo",
        "severidade_prevista"
    ]
].to_csv(
    PASTA_TABELAS / "base_previsoes_severidade.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nEtapa 5 concluída com sucesso.")