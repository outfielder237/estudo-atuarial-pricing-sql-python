import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

PASTA_DADOS = Path("../data")
PASTA_TABELAS = Path("../outputs/tabelas")
PASTA_FIGURAS = Path("../outputs/figuras")
ARQUIVO_BANCO = PASTA_DADOS / "pricing.db"

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

# 1. Leitura das bases
conexao = sqlite3.connect(ARQUIVO_BANCO)

apolices = pd.read_sql_query(
    '''
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
    ''',
    conexao
)

sinistros = pd.read_sql_query(
    '''
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
    ''',
    conexao
)

conexao.close()

print("\n================ BASES LIDAS ================")
print(f"Apólices: {apolices.shape}")
print(f"Sinistros: {sinistros.shape}")

# 2. Integridade
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

# 3. Base e modelo de frequência
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
    sinistros.groupby("id_apolice")
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
    base_freq["quantidade_sinistros"].fillna(0).astype(int)
)

base_freq["custo_total_sinistros"] = (
    base_freq["custo_total_sinistros"].fillna(0.0)
)

formula_freq = '''
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
'''

modelo_freq = smf.glm(
    formula=formula_freq,
    data=base_freq,
    family=sm.families.Poisson(),
    offset=base_freq["log_exposicao"]
).fit()

# 4. Modelo de severidade
sinistros["tipo_veiculo"] = pd.Categorical(
    sinistros["tipo_veiculo"],
    categories=["Hatch", "Sedan", "SUV", "Picape", "Utilitario"]
)

sinistros["nivel_cobertura"] = pd.Categorical(
    sinistros["nivel_cobertura"],
    categories=["Intermediaria", "Basica", "Completa"]
)

formula_sev = '''
valor_sinistro
~
C(tipo_veiculo, Treatment(reference="Hatch"))
+
C(nivel_cobertura, Treatment(reference="Intermediaria"))
+
idade_veiculo
'''

modelo_sev = smf.glm(
    formula=formula_sev,
    data=sinistros,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

# 5. Base final de pricing
base_pricing = base_freq.copy()

base_pricing["tipo_veiculo"] = pd.Categorical(
    base_pricing["tipo_veiculo"],
    categories=["Hatch", "Sedan", "SUV", "Picape", "Utilitario"]
)

base_pricing["nivel_cobertura"] = pd.Categorical(
    base_pricing["nivel_cobertura"],
    categories=["Intermediaria", "Basica", "Completa"]
)

# 6. Frequência prevista
base_pricing["sinistros_previstos_periodo"] = modelo_freq.predict(
    base_pricing,
    offset=base_pricing["log_exposicao"]
)

base_pricing["frequencia_prevista"] = (
    base_pricing["sinistros_previstos_periodo"] /
    base_pricing["exposicao"]
)

# 7. Severidade prevista
base_pricing["severidade_prevista"] = modelo_sev.predict(base_pricing)

# 8. Prêmio puro anual
base_pricing["premio_puro_anual"] = (
    base_pricing["frequencia_prevista"] *
    base_pricing["severidade_prevista"]
)

# 9. Custo esperado no período observado
base_pricing["custo_esperado_periodo"] = (
    base_pricing["premio_puro_anual"] *
    base_pricing["exposicao"]
)

# 10. Métricas individuais
base_pricing["lr_tecnico_individual"] = (
    base_pricing["premio_puro_anual"] /
    base_pricing["premio_vigente"]
)

base_pricing["adequacao_tarifaria"] = (
    base_pricing["premio_vigente"] /
    base_pricing["premio_puro_anual"]
)

# 11. Resumo integrado
custo_observado_total = base_pricing["custo_total_sinistros"].sum()
custo_esperado_total = base_pricing["custo_esperado_periodo"].sum()
premio_ganho_total = base_pricing["premio_ganho"].sum()
premio_vigente_total = base_pricing["premio_vigente"].sum()

sinistros_observados_total = base_pricing["quantidade_sinistros"].sum()
sinistros_previstos_total = base_pricing["sinistros_previstos_periodo"].sum()

oe_frequencia = sinistros_observados_total / sinistros_previstos_total
oe_custo = custo_observado_total / custo_esperado_total
lr_observado = custo_observado_total / premio_ganho_total
lr_tecnico = custo_esperado_total / premio_ganho_total

premio_puro_medio_anual = base_pricing["premio_puro_anual"].mean()
premio_vigente_medio = base_pricing["premio_vigente"].mean()
adequacao_media_razao = premio_vigente_medio / premio_puro_medio_anual

print("\n================ RESUMO INTEGRADO DE PRICING ================")
print(f"Sinistros observados: {sinistros_observados_total:,.0f}")
print(f"Sinistros previstos: {sinistros_previstos_total:,.2f}")
print(f"O/E frequência: {oe_frequencia:.4f}")
print(f"Custo observado total: R$ {custo_observado_total:,.2f}")
print(f"Custo esperado total: R$ {custo_esperado_total:,.2f}")
print(f"O/E custo: {oe_custo:.4f}")
print(f"Prêmio ganho total: R$ {premio_ganho_total:,.2f}")
print(f"Loss Ratio observado: {lr_observado:.2%}")
print(f"Loss Ratio técnico esperado: {lr_tecnico:.2%}")
print(f"Prêmio puro anual médio: R$ {premio_puro_medio_anual:,.2f}")
print(f"Prêmio vigente anual médio: R$ {premio_vigente_medio:,.2f}")
print(f"Razão prêmio vigente / prêmio puro médio: {adequacao_media_razao:.4f}")

pd.DataFrame({
    "metrica": [
        "sinistros_observados_total",
        "sinistros_previstos_total",
        "OE_frequencia",
        "custo_observado_total",
        "custo_esperado_total",
        "OE_custo",
        "premio_ganho_total",
        "premio_vigente_total",
        "loss_ratio_observado",
        "loss_ratio_tecnico",
        "premio_puro_medio_anual",
        "premio_vigente_medio",
        "razao_premio_vigente_premio_puro_medio"
    ],
    "valor": [
        sinistros_observados_total,
        sinistros_previstos_total,
        oe_frequencia,
        custo_observado_total,
        custo_esperado_total,
        oe_custo,
        premio_ganho_total,
        premio_vigente_total,
        lr_observado,
        lr_tecnico,
        premio_puro_medio_anual,
        premio_vigente_medio,
        adequacao_media_razao
    ]
}).to_csv(
    PASTA_TABELAS / "resumo_integrado_pricing.csv",
    index=False,
    encoding="utf-8-sig"
)

# 12. Distribuição do prêmio puro
resumo_distribuicao_pp = base_pricing["premio_puro_anual"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
)

print("\n================ DISTRIBUIÇÃO DO PRÊMIO PURO ================")
print(resumo_distribuicao_pp)

resumo_distribuicao_pp.to_csv(
    PASTA_TABELAS / "distribuicao_premio_puro.csv",
    encoding="utf-8-sig"
)

# 13. Faixa etária para análise
base_pricing["faixa_etaria_analise"] = pd.cut(
    base_pricing["idade_condutor"],
    bins=[17, 24, 34, 44, 54, 64, 80],
    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    include_lowest=True
)

# 14. Resumo por segmento
def resumo_por_segmento(df, coluna):
    resultado = (
        df.groupby(coluna, observed=False)
        .agg(
            apolices=("id_apolice", "count"),
            exposicao=("exposicao", "sum"),
            sinistros_observados=("quantidade_sinistros", "sum"),
            sinistros_previstos=("sinistros_previstos_periodo", "sum"),
            custo_observado=("custo_total_sinistros", "sum"),
            custo_esperado=("custo_esperado_periodo", "sum"),
            premio_ganho=("premio_ganho", "sum"),
            premio_vigente_medio=("premio_vigente", "mean"),
            premio_puro_medio=("premio_puro_anual", "mean"),
            severidade_prevista_media=("severidade_prevista", "mean")
        )
    )

    resultado["frequencia_observada"] = (
        resultado["sinistros_observados"] / resultado["exposicao"]
    )
    resultado["frequencia_prevista"] = (
        resultado["sinistros_previstos"] / resultado["exposicao"]
    )
    resultado["OE_frequencia"] = (
        resultado["sinistros_observados"] /
        resultado["sinistros_previstos"].replace(0, np.nan)
    )
    resultado["OE_custo"] = (
        resultado["custo_observado"] /
        resultado["custo_esperado"].replace(0, np.nan)
    )
    resultado["loss_ratio_observado"] = (
        resultado["custo_observado"] /
        resultado["premio_ganho"].replace(0, np.nan)
    )
    resultado["loss_ratio_tecnico"] = (
        resultado["custo_esperado"] /
        resultado["premio_ganho"].replace(0, np.nan)
    )
    resultado["adequacao_tarifaria_media"] = (
        resultado["premio_vigente_medio"] /
        resultado["premio_puro_medio"].replace(0, np.nan)
    )

    return resultado

resumo_regiao = resumo_por_segmento(base_pricing, "regiao")
resumo_tipo_uso = resumo_por_segmento(base_pricing, "tipo_uso")
resumo_tipo_veiculo = resumo_por_segmento(base_pricing, "tipo_veiculo")
resumo_cobertura = resumo_por_segmento(base_pricing, "nivel_cobertura")
resumo_bonus = resumo_por_segmento(base_pricing, "classe_bonus")
resumo_faixa_etaria = resumo_por_segmento(base_pricing, "faixa_etaria_analise")

for nome, tabela in {
    "regiao": resumo_regiao,
    "tipo_uso": resumo_tipo_uso,
    "tipo_veiculo": resumo_tipo_veiculo,
    "cobertura": resumo_cobertura,
    "bonus": resumo_bonus,
    "faixa_etaria": resumo_faixa_etaria
}.items():
    print(f"\n================ PRÊMIO PURO POR {nome.upper()} ================")
    print(tabela)

    tabela.to_csv(
        PASTA_TABELAS / f"premio_puro_por_{nome}.csv",
        encoding="utf-8-sig"
    )

# 15. Exportação da base completa
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
    "sinistros_previstos_periodo",
    "frequencia_prevista",
    "severidade_prevista",
    "premio_puro_anual",
    "custo_esperado_periodo",
    "lr_tecnico_individual",
    "adequacao_tarifaria"
]

base_pricing[colunas_exportacao].to_csv(
    PASTA_TABELAS / "base_pricing_premio_puro.csv",
    index=False,
    encoding="utf-8-sig"
)

# 16. Gráficos
plt.figure(figsize=(9, 5))
plt.hist(base_pricing["premio_puro_anual"], bins=50)
plt.xlabel("Prêmio puro anual (R$)")
plt.ylabel("Quantidade de apólices")
plt.title("Distribuição do prêmio puro anual")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "distribuicao_premio_puro.png", dpi=150)
plt.show()

def grafico_premio_puro_segmento(tabela, titulo, nome_arquivo):
    categorias = tabela.index.astype(str)

    plt.figure(figsize=(10, 5))
    plt.bar(categorias, tabela["premio_puro_medio"])
    plt.xlabel("Segmento")
    plt.ylabel("Prêmio puro médio anual (R$)")
    plt.title(titulo)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150)
    plt.show()

grafico_premio_puro_segmento(
    resumo_tipo_veiculo,
    "Prêmio puro médio por tipo de veículo",
    "premio_puro_tipo_veiculo.png"
)

grafico_premio_puro_segmento(
    resumo_tipo_uso,
    "Prêmio puro médio por tipo de uso",
    "premio_puro_tipo_uso.png"
)

grafico_premio_puro_segmento(
    resumo_bonus,
    "Prêmio puro médio por classe de bônus",
    "premio_puro_bonus.png"
)

grafico_premio_puro_segmento(
    resumo_faixa_etaria,
    "Prêmio puro médio por faixa etária",
    "premio_puro_faixa_etaria.png"
)

plt.figure(figsize=(8, 8))
plt.scatter(
    base_pricing["premio_vigente"],
    base_pricing["premio_puro_anual"],
    alpha=0.20,
    s=10
)

limite = max(
    base_pricing["premio_vigente"].max(),
    base_pricing["premio_puro_anual"].max()
)

plt.plot([0, limite], [0, limite], linestyle="--")
plt.xlabel("Prêmio vigente anual (R$)")
plt.ylabel("Prêmio puro anual indicado (R$)")
plt.title("Prêmio vigente × prêmio puro indicado")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "premio_vigente_vs_puro.png", dpi=150)
plt.show()

def grafico_lr_segmento(tabela, titulo, nome_arquivo):
    categorias = tabela.index.astype(str)
    x = np.arange(len(categorias))
    largura = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - largura / 2,
        tabela["loss_ratio_observado"],
        width=largura,
        label="Observado"
    )
    plt.bar(
        x + largura / 2,
        tabela["loss_ratio_tecnico"],
        width=largura,
        label="Técnico esperado"
    )
    plt.xticks(x, categorias, rotation=45)
    plt.ylabel("Loss Ratio")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150)
    plt.show()

grafico_lr_segmento(
    resumo_tipo_veiculo,
    "Loss Ratio observado × técnico por tipo de veículo",
    "lr_observado_vs_tecnico_tipo_veiculo.png"
)

grafico_lr_segmento(
    resumo_regiao,
    "Loss Ratio observado × técnico por região",
    "lr_observado_vs_tecnico_regiao.png"
)

print("\nEtapa 6 concluída com sucesso.")