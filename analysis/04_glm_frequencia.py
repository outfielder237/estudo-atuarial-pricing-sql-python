import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

PASTA_DADOS = Path("../data")
PASTA_TABELAS = Path("../outputs/tabelas")
PASTA_FIGURAS = Path("../outputs/figuras")
ARQUIVO_BANCO = PASTA_DADOS / "pricing.db"

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

conexao = sqlite3.connect(ARQUIVO_BANCO)
base = pd.read_sql_query(
    """
    SELECT *
    FROM vw_base_modelagem
    ORDER BY id_apolice;
    """,
    conexao
)
conexao.close()

print("\nDimensão da base:")
print(base.shape)
print("\nPrimeiras linhas:")
print(base.head())

if (base["exposicao"] <= 0).any():
    raise ValueError("Existem exposições menores ou iguais a zero.")

base["log_exposicao"] = np.log(base["exposicao"])

base["faixa_idade_modelo"] = pd.cut(
    base["idade_condutor"],
    bins=[17, 24, 65, 80],
    labels=["18-24", "25-65", "66+"],
    include_lowest=True
)

base["faixa_idade_modelo"] = pd.Categorical(
    base["faixa_idade_modelo"],
    categories=["25-65", "18-24", "66+"]
)

base["regiao"] = pd.Categorical(
    base["regiao"],
    categories=["Sul", "Sudeste", "Nordeste", "Centro-Oeste", "Norte"]
)

base["tipo_uso"] = pd.Categorical(
    base["tipo_uso"],
    categories=["Particular", "Comercial"]
)

formula = """
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

modelo_nulo = smf.glm(
    formula="quantidade_sinistros ~ 1",
    data=base,
    family=sm.families.Poisson(),
    offset=base["log_exposicao"]
).fit()

modelo_poisson = smf.glm(
    formula=formula,
    data=base,
    family=sm.families.Poisson(),
    offset=base["log_exposicao"]
).fit()

print("\n================ RESUMO DO GLM POISSON ================")
print(modelo_poisson.summary())

intervalos = modelo_poisson.conf_int()

tabela_coeficientes = pd.DataFrame({
    "coeficiente": modelo_poisson.params,
    "erro_padrao": modelo_poisson.bse,
    "p_valor": modelo_poisson.pvalues,
    "relatividade": np.exp(modelo_poisson.params),
    "relatividade_limite_inferior": np.exp(intervalos[0]),
    "relatividade_limite_superior": np.exp(intervalos[1])
})

print("\n================ COEFICIENTES E RELATIVIDADES ================")
print(tabela_coeficientes)

tabela_coeficientes.to_csv(
    PASTA_TABELAS / "coeficientes_glm_frequencia.csv",
    encoding="utf-8-sig"
)

coeficientes_verdadeiros = {
    'C(faixa_idade_modelo, Treatment(reference="25-65"))[T.18-24]': 0.35,
    'C(faixa_idade_modelo, Treatment(reference="25-65"))[T.66+]': 0.20,
    'idade_veiculo': 0.012,
    'C(regiao, Treatment(reference="Sul"))[T.Sudeste]': 0.10,
    'C(regiao, Treatment(reference="Sul"))[T.Nordeste]': -0.05,
    'C(regiao, Treatment(reference="Sul"))[T.Centro-Oeste]': 0.08,
    'C(regiao, Treatment(reference="Sul"))[T.Norte]': 0.03,
    'C(tipo_uso, Treatment(reference="Particular"))[T.Comercial]': 0.30,
    'classe_bonus': -0.055
}

linhas = []
for termo, beta_verdadeiro in coeficientes_verdadeiros.items():
    beta_estimado = modelo_poisson.params.get(termo, np.nan)
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
    PASTA_TABELAS / "comparacao_coeficientes_verdadeiros_estimados.csv",
    index=False,
    encoding="utf-8-sig"
)

dispersao_deviance = modelo_poisson.deviance / modelo_poisson.df_resid
dispersao_pearson = modelo_poisson.pearson_chi2 / modelo_poisson.df_resid

print("\n================ DIAGNÓSTICO DE DISPERSÃO ================")
print(f"Deviance / graus de liberdade: {dispersao_deviance:.4f}")
print(f"Pearson Chi² / graus de liberdade: {dispersao_pearson:.4f}")

if dispersao_pearson > 1.20:
    print("Há sinal relevante de sobredispersão.")
elif dispersao_pearson < 0.80:
    print("Há sinal relevante de subdispersão.")
else:
    print("A dispersão está próxima da estrutura esperada para Poisson.")

reducao_deviance = modelo_nulo.deviance - modelo_poisson.deviance

print("\n================ MODELO NULO × MODELO COMPLETO ================")
print(f"AIC modelo nulo: {modelo_nulo.aic:.4f}")
print(f"AIC modelo completo: {modelo_poisson.aic:.4f}")
print(f"Deviance modelo nulo: {modelo_nulo.deviance:.4f}")
print(f"Deviance modelo completo: {modelo_poisson.deviance:.4f}")
print(f"Redução de Deviance: {reducao_deviance:.4f}")

pd.DataFrame({
    "modelo": ["Nulo", "Poisson completo"],
    "AIC": [modelo_nulo.aic, modelo_poisson.aic],
    "deviance": [modelo_nulo.deviance, modelo_poisson.deviance],
    "pearson_chi2": [modelo_nulo.pearson_chi2, modelo_poisson.pearson_chi2],
    "df_resid": [modelo_nulo.df_resid, modelo_poisson.df_resid]
}).to_csv(
    PASTA_TABELAS / "comparacao_modelos_frequencia.csv",
    index=False,
    encoding="utf-8-sig"
)

base["sinistros_previstos"] = modelo_poisson.predict(
    base,
    offset=base["log_exposicao"]
)

base["frequencia_prevista"] = (
    base["sinistros_previstos"] / base["exposicao"]
)

frequencia_observada = base["quantidade_sinistros"].sum() / base["exposicao"].sum()
frequencia_prevista = base["sinistros_previstos"].sum() / base["exposicao"].sum()

print("\n================ OBSERVADO × PREVISTO — TOTAL ================")
print(f"Frequência observada: {frequencia_observada:.6f}")
print(f"Frequência prevista: {frequencia_prevista:.6f}")

def validar_segmento(df, coluna):
    resultado = (
        df.groupby(coluna, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            exposicao=("exposicao", "sum"),
            sinistros_observados=("quantidade_sinistros", "sum"),
            sinistros_previstos=("sinistros_previstos", "sum")
        )
    )
    resultado["frequencia_observada"] = (
        resultado["sinistros_observados"] / resultado["exposicao"]
    )
    resultado["frequencia_prevista"] = (
        resultado["sinistros_previstos"] / resultado["exposicao"]
    )
    resultado["OE"] = (
        resultado["sinistros_observados"] /
        resultado["sinistros_previstos"].replace(0, np.nan)
    )
    return resultado

validacao_uso = validar_segmento(base, "tipo_uso")
validacao_regiao = validar_segmento(base, "regiao")
validacao_idade = validar_segmento(base, "faixa_idade_modelo")
validacao_bonus = validar_segmento(base, "classe_bonus")

print("\n================ VALIDAÇÃO — TIPO DE USO ================")
print(validacao_uso)
print("\n================ VALIDAÇÃO — REGIÃO ================")
print(validacao_regiao)
print("\n================ VALIDAÇÃO — FAIXA ETÁRIA ================")
print(validacao_idade)
print("\n================ VALIDAÇÃO — CLASSE DE BÔNUS ================")
print(validacao_bonus)

validacao_uso.to_csv(PASTA_TABELAS / "validacao_tipo_uso.csv", encoding="utf-8-sig")
validacao_regiao.to_csv(PASTA_TABELAS / "validacao_regiao.csv", encoding="utf-8-sig")
validacao_idade.to_csv(PASTA_TABELAS / "validacao_idade.csv", encoding="utf-8-sig")
validacao_bonus.to_csv(PASTA_TABELAS / "validacao_bonus.csv", encoding="utf-8-sig")

treino, teste = train_test_split(
    base,
    test_size=0.20,
    random_state=20260906
)

treino = treino.copy()
teste = teste.copy()

modelo_treino = smf.glm(
    formula=formula,
    data=treino,
    family=sm.families.Poisson(),
    offset=treino["log_exposicao"]
).fit()

teste["sinistros_previstos"] = modelo_treino.predict(
    teste,
    offset=teste["log_exposicao"]
)

obs_teste = teste["quantidade_sinistros"].sum()
prev_teste = teste["sinistros_previstos"].sum()
oe_teste = obs_teste / prev_teste

freq_obs_teste = obs_teste / teste["exposicao"].sum()
freq_prev_teste = prev_teste / teste["exposicao"].sum()

print("\n================ VALIDAÇÃO FORA DA AMOSTRA ================")
print(f"Apólices no treino: {len(treino):,}")
print(f"Apólices no teste: {len(teste):,}")
print(f"Sinistros observados no teste: {obs_teste:.0f}")
print(f"Sinistros previstos no teste: {prev_teste:.2f}")
print(f"Frequência observada no teste: {freq_obs_teste:.6f}")
print(f"Frequência prevista no teste: {freq_prev_teste:.6f}")
print(f"Razão O/E no teste: {oe_teste:.4f}")

pd.DataFrame({
    "metrica": [
        "apolices_treino",
        "apolices_teste",
        "sinistros_observados_teste",
        "sinistros_previstos_teste",
        "frequencia_observada_teste",
        "frequencia_prevista_teste",
        "OE_teste"
    ],
    "valor": [
        len(treino),
        len(teste),
        obs_teste,
        prev_teste,
        freq_obs_teste,
        freq_prev_teste,
        oe_teste
    ]
}).to_csv(
    PASTA_TABELAS / "validacao_teste_glm_frequencia.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 18A. Validação segmentada fora da amostra
# ------------------------------------------------------------
#
# Diferentemente da validação "in-sample" feita anteriormente,
# aqui avaliamos o modelo em apólices que não participaram do
# ajuste do GLM de treino.
#
# Isso permite verificar se a calibração se mantém por segmento,
# em vez de olhar apenas o O/E agregado do conjunto de teste.

def validar_segmento_teste(df, coluna):
    resultado = (
        df
        .groupby(coluna, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            exposicao=("exposicao", "sum"),
            sinistros_observados=("quantidade_sinistros", "sum"),
            sinistros_previstos=("sinistros_previstos", "sum")
        )
    )

    resultado["frequencia_observada"] = (
        resultado["sinistros_observados"]
        /
        resultado["exposicao"]
    )

    resultado["frequencia_prevista"] = (
        resultado["sinistros_previstos"]
        /
        resultado["exposicao"]
    )

    resultado["OE"] = (
        resultado["sinistros_observados"]
        /
        resultado["sinistros_previstos"].replace(0, np.nan)
    )

    resultado["erro_relativo_previsao"] = (
        resultado["sinistros_previstos"]
        -
        resultado["sinistros_observados"]
    ) / resultado["sinistros_observados"].replace(0, np.nan)

    return resultado


validacao_teste_uso = validar_segmento_teste(
    teste,
    "tipo_uso"
)

validacao_teste_regiao = validar_segmento_teste(
    teste,
    "regiao"
)

validacao_teste_idade = validar_segmento_teste(
    teste,
    "faixa_idade_modelo"
)

validacao_teste_bonus = validar_segmento_teste(
    teste,
    "classe_bonus"
)

print(
    "\\n================ VALIDAÇÃO FORA DA AMOSTRA — TIPO DE USO ================"
)
print(validacao_teste_uso)

print(
    "\\n================ VALIDAÇÃO FORA DA AMOSTRA — REGIÃO ================"
)
print(validacao_teste_regiao)

print(
    "\\n================ VALIDAÇÃO FORA DA AMOSTRA — FAIXA ETÁRIA ================"
)
print(validacao_teste_idade)

print(
    "\\n================ VALIDAÇÃO FORA DA AMOSTRA — CLASSE DE BÔNUS ================"
)
print(validacao_teste_bonus)

# Exportação das validações segmentadas fora da amostra

validacao_teste_uso.to_csv(
    PASTA_TABELAS / "validacao_teste_tipo_uso.csv",
    encoding="utf-8-sig"
)

validacao_teste_regiao.to_csv(
    PASTA_TABELAS / "validacao_teste_regiao.csv",
    encoding="utf-8-sig"
)

validacao_teste_idade.to_csv(
    PASTA_TABELAS / "validacao_teste_idade.csv",
    encoding="utf-8-sig"
)

validacao_teste_bonus.to_csv(
    PASTA_TABELAS / "validacao_teste_bonus.csv",
    encoding="utf-8-sig"
)

# Resumo consolidado dos O/E fora da amostra

def consolidar_oe(tabela, nome_segmento):
    temp = tabela.reset_index().copy()

    primeira_coluna = temp.columns[0]
    temp = temp.rename(
        columns={primeira_coluna: "categoria"}
    )

    temp.insert(
        0,
        "segmento",
        nome_segmento
    )

    return temp[
        [
            "segmento",
            "categoria",
            "apolices",
            "exposicao",
            "sinistros_observados",
            "sinistros_previstos",
            "frequencia_observada",
            "frequencia_prevista",
            "OE",
            "erro_relativo_previsao"
        ]
    ]


resumo_validacao_segmentada_teste = pd.concat(
    [
        consolidar_oe(
            validacao_teste_uso,
            "tipo_uso"
        ),
        consolidar_oe(
            validacao_teste_regiao,
            "regiao"
        ),
        consolidar_oe(
            validacao_teste_idade,
            "faixa_idade_modelo"
        ),
        consolidar_oe(
            validacao_teste_bonus,
            "classe_bonus"
        )
    ],
    ignore_index=True
)

resumo_validacao_segmentada_teste.to_csv(
    PASTA_TABELAS / "validacao_segmentada_teste_consolidada.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_plot = tabela_coeficientes.drop(index="Intercept", errors="ignore").copy()

plt.figure(figsize=(10, 6))
plt.barh(
    tabela_plot.index.astype(str),
    tabela_plot["relatividade"]
)
plt.axvline(1.0, linestyle="--")
plt.xlabel("Relatividade estimada")
plt.ylabel("Variável")
plt.title("Relatividades estimadas — GLM de frequência")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "relatividades_frequencia.png", dpi=150)
plt.show()

def grafico_observado_previsto(tabela, titulo, nome_arquivo):
    categorias = tabela.index.astype(str)
    x = np.arange(len(categorias))
    largura = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - largura / 2,
        tabela["frequencia_observada"],
        width=largura,
        label="Observada"
    )
    plt.bar(
        x + largura / 2,
        tabela["frequencia_prevista"],
        width=largura,
        label="Prevista"
    )
    plt.xticks(x, categorias, rotation=45)
    plt.ylabel("Frequência anual")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150)
    plt.show()

grafico_observado_previsto(
    validacao_uso,
    "Frequência observada × prevista por tipo de uso",
    "observado_previsto_tipo_uso.png"
)

grafico_observado_previsto(
    validacao_regiao,
    "Frequência observada × prevista por região",
    "observado_previsto_regiao.png"
)

grafico_observado_previsto(
    validacao_idade,
    "Frequência observada × prevista por faixa etária",
    "observado_previsto_idade.png"
)

grafico_observado_previsto(
    validacao_bonus,
    "Frequência observada × prevista por classe de bônus",
    "observado_previsto_bonus.png"
)


# ------------------------------------------------------------
# 20A. Gráficos de validação fora da amostra
# ------------------------------------------------------------

grafico_observado_previsto(
    validacao_teste_uso,
    "Teste — frequência observada × prevista por tipo de uso",
    "teste_observado_previsto_tipo_uso.png"
)

grafico_observado_previsto(
    validacao_teste_regiao,
    "Teste — frequência observada × prevista por região",
    "teste_observado_previsto_regiao.png"
)

grafico_observado_previsto(
    validacao_teste_idade,
    "Teste — frequência observada × prevista por faixa etária",
    "teste_observado_previsto_idade.png"
)

grafico_observado_previsto(
    validacao_teste_bonus,
    "Teste — frequência observada × prevista por classe de bônus",
    "teste_observado_previsto_bonus.png"
)

colunas_exportacao = [
    "id_apolice",
    "exposicao",
    "idade_condutor",
    "faixa_idade_modelo",
    "idade_veiculo",
    "regiao",
    "tipo_uso",
    "classe_bonus",
    "quantidade_sinistros",
    "sinistros_previstos",
    "frequencia_prevista"
]

base[colunas_exportacao].to_csv(
    PASTA_TABELAS / "base_previsoes_frequencia.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nEtapa 4 concluída com sucesso.")