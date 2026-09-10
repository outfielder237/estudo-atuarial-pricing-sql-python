import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ETAPA 3 — ANÁLISE EXPLORATÓRIA DA BASE DE MODELAGEM
# Projeto: Pricing + Capital com SQL e Python
# Contexto: Seguro Automóvel — carteira sintética
# ============================================================

# 1. Caminhos e conexão
PASTA_DADOS = Path('../data')
PASTA_TABELAS = Path('../outputs/tabelas')
PASTA_FIGURAS = Path('../outputs/figuras')
ARQUIVO_BANCO = PASTA_DADOS / 'pricing.db'

PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

conexao = sqlite3.connect(ARQUIVO_BANCO)

# 2. Leitura da VIEW atuarial
base = pd.read_sql_query(
    '''
    SELECT *
    FROM vw_base_modelagem
    ORDER BY id_apolice;
    ''',
    conexao
)

print('\nPrimeiras linhas da base:')
print(base.head())
print('\nDimensão da base:')
print(base.shape)
print('\nInformações da base:')
base.info()

# 3. Métricas auxiliares
base['frequencia_observada'] = base['quantidade_sinistros'] / base['exposicao']
base['tem_sinistro'] = (base['quantidade_sinistros'] > 0).astype(int)
base['severidade_observada'] = np.where(
    base['quantidade_sinistros'] > 0,
    base['custo_total_sinistros'] / base['quantidade_sinistros'],
    np.nan
)
base['custo_por_exposicao'] = base['custo_total_sinistros'] / base['exposicao']

# 4. Resumo geral
exposicao_total = base['exposicao'].sum()
sinistros_total = base['quantidade_sinistros'].sum()
custo_total = base['custo_total_sinistros'].sum()
premio_total = base['premio_ganho'].sum()

frequencia = sinistros_total / exposicao_total
severidade = custo_total / sinistros_total
loss_ratio = custo_total / premio_total

print('\n================ RESUMO GERAL ================')
print(f'Exposição total: {exposicao_total:,.2f}')
print(f'Sinistros: {sinistros_total:,.0f}')
print(f'Frequência: {frequencia:.2%}')
print(f'Severidade média: R$ {severidade:,.2f}')
print(f'Prêmio ganho: R$ {premio_total:,.2f}')
print(f'Custo total de sinistros: R$ {custo_total:,.2f}')
print(f'Loss Ratio: {loss_ratio:.2%}')

# 5. Distribuição da contagem
distribuicao_contagem = base['quantidade_sinistros'].value_counts().sort_index()
proporcao_contagem = base['quantidade_sinistros'].value_counts(normalize=True).sort_index()

print('\nDistribuição da contagem de sinistros:')
print(distribuicao_contagem)
print('\nProporção da contagem de sinistros:')
print(proporcao_contagem)

# 6. Diagnóstico inicial de sobredispersão
media_contagem = base['quantidade_sinistros'].mean()
variancia_contagem = base['quantidade_sinistros'].var()
razao_var_media = variancia_contagem / media_contagem

print('\n================ DISPERSÃO ================')
print(f'Média da contagem: {media_contagem:.6f}')
print(f'Variância da contagem: {variancia_contagem:.6f}')
print(f'Razão variância/média: {razao_var_media:.4f}')

if razao_var_media > 1.20:
    print('Sinal inicial de sobredispersão.')
elif razao_var_media < 0.80:
    print('Sinal inicial de subdispersão.')
else:
    print('Dispersão bruta próxima da estrutura de Poisson.')

# 7. Função auxiliar
def adicionar_metricas_atuariais(df):
    df = df.copy()
    df['frequencia'] = df['sinistros'] / df['exposicao']
    df['severidade'] = df['custo'] / df['sinistros'].replace(0, np.nan)
    df['loss_ratio'] = df['custo'] / df['premio'].replace(0, np.nan)
    return df

# 8. Resumo por tipo de uso
resumo_uso = base.groupby('tipo_uso').agg(
    apolices=('id_apolice', 'count'),
    exposicao=('exposicao', 'sum'),
    sinistros=('quantidade_sinistros', 'sum'),
    custo=('custo_total_sinistros', 'sum'),
    premio=('premio_ganho', 'sum')
)
resumo_uso = adicionar_metricas_atuariais(resumo_uso)
print('\n================ TIPO DE USO ================')
print(resumo_uso)

# 9. Resumo por classe de bônus
resumo_bonus = base.groupby('classe_bonus').agg(
    apolices=('id_apolice', 'count'),
    exposicao=('exposicao', 'sum'),
    sinistros=('quantidade_sinistros', 'sum'),
    custo=('custo_total_sinistros', 'sum'),
    premio=('premio_ganho', 'sum')
)
resumo_bonus = adicionar_metricas_atuariais(resumo_bonus)
print('\n================ CLASSE DE BÔNUS ================')
print(resumo_bonus)

# 10. Resumo por tipo de veículo
resumo_veiculo = base.groupby('tipo_veiculo').agg(
    apolices=('id_apolice', 'count'),
    exposicao=('exposicao', 'sum'),
    sinistros=('quantidade_sinistros', 'sum'),
    custo=('custo_total_sinistros', 'sum'),
    premio=('premio_ganho', 'sum')
)
resumo_veiculo = adicionar_metricas_atuariais(resumo_veiculo)
print('\n================ TIPO DE VEÍCULO ================')
print(resumo_veiculo)

# 11. Faixas etárias
base['faixa_idade_condutor'] = pd.cut(
    base['idade_condutor'],
    bins=[17, 24, 34, 44, 54, 64, 80],
    labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

resumo_idade = base.groupby('faixa_idade_condutor', observed=False).agg(
    apolices=('id_apolice', 'count'),
    exposicao=('exposicao', 'sum'),
    sinistros=('quantidade_sinistros', 'sum'),
    custo=('custo_total_sinistros', 'sum'),
    premio=('premio_ganho', 'sum')
)
resumo_idade = adicionar_metricas_atuariais(resumo_idade)
print('\n================ FAIXA ETÁRIA ================')
print(resumo_idade)

# 12. Análise da severidade
base_severidade = base[base['quantidade_sinistros'] > 0].copy()
print('\n================ SEVERIDADE ================')
print(
    base_severidade['severidade_observada'].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# 13. Resumo por região
resumo_regiao = base.groupby('regiao').agg(
    apolices=('id_apolice', 'count'),
    exposicao=('exposicao', 'sum'),
    sinistros=('quantidade_sinistros', 'sum'),
    custo=('custo_total_sinistros', 'sum'),
    premio=('premio_ganho', 'sum')
)
resumo_regiao = adicionar_metricas_atuariais(resumo_regiao)
resumo_regiao['premio_por_exposicao'] = resumo_regiao['premio'] / resumo_regiao['exposicao']
resumo_regiao['custo_por_exposicao'] = resumo_regiao['custo'] / resumo_regiao['exposicao']
print('\n================ REGIÃO ================')
print(resumo_regiao)

# 14. Gráfico — Distribuição da severidade
plt.figure(figsize=(9, 5))
plt.hist(base_severidade['severidade_observada'], bins=50)
plt.xlabel('Severidade observada (R$)')
plt.ylabel('Frequência')
plt.title('Distribuição da severidade observada')
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / 'distribuicao_severidade.png', dpi=150)
plt.show()

# 15. Gráfico — Frequência por classe de bônus
plt.figure(figsize=(9, 5))
plt.plot(resumo_bonus.index, resumo_bonus['frequencia'], marker='o')
plt.xlabel('Classe de bônus')
plt.ylabel('Frequência anual')
plt.title('Frequência de sinistros por classe de bônus')
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / 'frequencia_classe_bonus.png', dpi=150)
plt.show()

# 16. Gráfico — Frequência por faixa etária
plt.figure(figsize=(9, 5))
plt.plot(resumo_idade.index.astype(str), resumo_idade['frequencia'], marker='o')
plt.xlabel('Faixa etária do condutor')
plt.ylabel('Frequência anual')
plt.title('Frequência de sinistros por faixa etária')
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / 'frequencia_faixa_etaria.png', dpi=150)
plt.show()

# 17. Gráfico — Severidade por tipo de veículo
plt.figure(figsize=(9, 5))
plt.bar(resumo_veiculo.index, resumo_veiculo['severidade'])
plt.xlabel('Tipo de veículo')
plt.ylabel('Severidade média (R$)')
plt.title('Severidade média por tipo de veículo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / 'severidade_tipo_veiculo.png', dpi=150)
plt.show()

# 18. Gráfico — Prêmio versus custo por exposição por região
x = np.arange(len(resumo_regiao.index))
largura = 0.35

plt.figure(figsize=(10, 5))
plt.bar(
    x - largura / 2,
    resumo_regiao['premio_por_exposicao'],
    width=largura,
    label='Prêmio por exposição'
)
plt.bar(
    x + largura / 2,
    resumo_regiao['custo_por_exposicao'],
    width=largura,
    label='Custo por exposição'
)
plt.xticks(x, resumo_regiao.index, rotation=45)
plt.ylabel('R$ por unidade de exposição')
plt.title('Prêmio versus custo por exposição por região')
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / 'premio_vs_custo_regiao.png', dpi=150)
plt.show()

# 19. Exportação das tabelas
resumo_uso.to_csv(PASTA_TABELAS / 'resumo_tipo_uso.csv', encoding='utf-8-sig')
resumo_bonus.to_csv(PASTA_TABELAS / 'resumo_classe_bonus.csv', encoding='utf-8-sig')
resumo_veiculo.to_csv(PASTA_TABELAS / 'resumo_tipo_veiculo.csv', encoding='utf-8-sig')
resumo_idade.to_csv(PASTA_TABELAS / 'resumo_faixa_idade.csv', encoding='utf-8-sig')
resumo_regiao.to_csv(PASTA_TABELAS / 'resumo_regiao.csv', encoding='utf-8-sig')
distribuicao_contagem.to_csv(
    PASTA_TABELAS / 'distribuicao_contagem_sinistros.csv',
    encoding='utf-8-sig'
)
proporcao_contagem.to_csv(
    PASTA_TABELAS / 'proporcao_contagem_sinistros.csv',
    encoding='utf-8-sig'
)

# 20. Encerramento
conexao.close()
print('\nEtapa 3 concluída com sucesso.')