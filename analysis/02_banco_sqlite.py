import sqlite3
import pandas as pd
from pathlib import Path

# ============================================================
# 1. Caminhos
# ============================================================

PASTA_DADOS = Path("../data")
ARQUIVO_APOLICES = PASTA_DADOS / "apolices.csv"
ARQUIVO_SINISTROS = PASTA_DADOS / "sinistros.csv"
ARQUIVO_BANCO = PASTA_DADOS / "pricing.db"

# ============================================================
# 2. Leitura dos CSVs
# ============================================================

apolices = pd.read_csv(ARQUIVO_APOLICES)
sinistros = pd.read_csv(ARQUIVO_SINISTROS)

print("\nAmostra de apólices:")
print(apolices.head())

print("\nAmostra de sinistros:")
print(sinistros.head())

print("\nDimensões:")
print("Apólices:", apolices.shape)
print("Sinistros:", sinistros.shape)

# ============================================================
# 3. Conexão com SQLite
# ============================================================

conexao = sqlite3.connect(ARQUIVO_BANCO)
conexao.execute("PRAGMA foreign_keys = ON;")

# ============================================================
# 4. Limpeza do banco anterior
# ============================================================

conexao.execute("DROP VIEW IF EXISTS vw_base_modelagem;")
conexao.execute("DROP TABLE IF EXISTS sinistros;")
conexao.execute("DROP TABLE IF EXISTS apolices;")
conexao.commit()

# ============================================================
# 5. Criação das tabelas
# ============================================================

sql_apolices = '''
CREATE TABLE apolices (
    id_apolice INTEGER PRIMARY KEY,
    exposicao REAL NOT NULL,
    idade_condutor INTEGER NOT NULL,
    idade_veiculo INTEGER NOT NULL,
    regiao TEXT NOT NULL,
    tipo_veiculo TEXT NOT NULL,
    tipo_uso TEXT NOT NULL,
    nivel_cobertura TEXT NOT NULL,
    classe_bonus INTEGER NOT NULL,
    premio_vigente REAL NOT NULL,
    premio_ganho REAL NOT NULL
);
'''

sql_sinistros = '''
CREATE TABLE sinistros (
    id_sinistro INTEGER PRIMARY KEY,
    id_apolice INTEGER NOT NULL,
    valor_sinistro REAL NOT NULL,
    FOREIGN KEY (id_apolice)
        REFERENCES apolices(id_apolice)
);
'''

conexao.execute(sql_apolices)
conexao.execute(sql_sinistros)
conexao.commit()

# ============================================================
# 6. Carga dos dados
# ============================================================

apolices.to_sql("apolices", conexao, if_exists="append", index=False)
sinistros.to_sql("sinistros", conexao, if_exists="append", index=False)
conexao.commit()

# ============================================================
# 7. Índices
# ============================================================

conexao.execute('''
CREATE INDEX IF NOT EXISTS idx_sinistros_id_apolice
ON sinistros(id_apolice);
''')

conexao.execute('''
CREATE INDEX IF NOT EXISTS idx_apolices_regiao
ON apolices(regiao);
''')

conexao.execute('''
CREATE INDEX IF NOT EXISTS idx_apolices_tipo_veiculo
ON apolices(tipo_veiculo);
''')

conexao.commit()

# ============================================================
# 8. Validação das quantidades
# ============================================================

query = '''
SELECT COUNT(*) AS quantidade_apolices
FROM apolices;
'''
resultado_apolices = pd.read_sql_query(query, conexao)
print("\nQuantidade de apólices:")
print(resultado_apolices)

query = '''
SELECT COUNT(*) AS quantidade_sinistros
FROM sinistros;
'''
resultado_sinistros = pd.read_sql_query(query, conexao)
print("\nQuantidade de sinistros:")
print(resultado_sinistros)

# ============================================================
# 9. Validação de integridade referencial
# ============================================================

query = '''
SELECT COUNT(*) AS sinistros_orfaos
FROM sinistros s
LEFT JOIN apolices a
    ON s.id_apolice = a.id_apolice
WHERE a.id_apolice IS NULL;
'''
resultado_orfaos = pd.read_sql_query(query, conexao)
print("\nSinistros órfãos:")
print(resultado_orfaos)

# ============================================================
# 10. Construção da base atuarial de modelagem
# ============================================================

query = '''
SELECT
    a.id_apolice,
    a.exposicao,
    a.idade_condutor,
    a.idade_veiculo,
    a.regiao,
    a.tipo_veiculo,
    a.tipo_uso,
    a.nivel_cobertura,
    a.classe_bonus,
    a.premio_vigente,
    a.premio_ganho,
    COUNT(s.id_sinistro) AS quantidade_sinistros,
    COALESCE(SUM(s.valor_sinistro), 0) AS custo_total_sinistros
FROM apolices a
LEFT JOIN sinistros s
    ON a.id_apolice = s.id_apolice
GROUP BY
    a.id_apolice,
    a.exposicao,
    a.idade_condutor,
    a.idade_veiculo,
    a.regiao,
    a.tipo_veiculo,
    a.tipo_uso,
    a.nivel_cobertura,
    a.classe_bonus,
    a.premio_vigente,
    a.premio_ganho;
'''

base_modelagem = pd.read_sql_query(query, conexao)

print("\nPrimeiras linhas da base de modelagem:")
print(base_modelagem.head())

print("\nDimensão da base de modelagem:")
print(base_modelagem.shape)

print("\nTotal de sinistros na base de modelagem:")
print(base_modelagem["quantidade_sinistros"].sum())

print("\nCusto total de sinistros na base de modelagem:")
print(base_modelagem["custo_total_sinistros"].sum())

# ============================================================
# 11. Frequência atuarial
# ============================================================

query = '''
SELECT
    SUM(quantidade_sinistros) * 1.0 / SUM(exposicao) AS frequencia
FROM (
    SELECT
        a.id_apolice,
        a.exposicao,
        COUNT(s.id_sinistro) AS quantidade_sinistros
    FROM apolices a
    LEFT JOIN sinistros s
        ON a.id_apolice = s.id_apolice
    GROUP BY a.id_apolice, a.exposicao
);
'''
resultado_frequencia = pd.read_sql_query(query, conexao)
print("\nFrequência atuarial:")
print(resultado_frequencia)

# ============================================================
# 12. Severidade média
# ============================================================

query = '''
SELECT
    SUM(valor_sinistro) * 1.0 / COUNT(*) AS severidade_media
FROM sinistros;
'''
resultado_severidade = pd.read_sql_query(query, conexao)
print("\nSeveridade média:")
print(resultado_severidade)

# ============================================================
# 13. Loss Ratio
# ============================================================

query = '''
SELECT
    (
        SELECT SUM(valor_sinistro)
        FROM sinistros
    ) * 1.0
    /
    (
        SELECT SUM(premio_ganho)
        FROM apolices
    )
    AS loss_ratio;
'''
resultado_loss_ratio = pd.read_sql_query(query, conexao)
print("\nLoss Ratio:")
print(resultado_loss_ratio)

# ============================================================
# 14. Resumo atuarial por região
# ============================================================

query = '''
WITH sinistros_por_apolice AS (
    SELECT
        id_apolice,
        COUNT(*) AS quantidade_sinistros,
        SUM(valor_sinistro) AS custo_sinistros
    FROM sinistros
    GROUP BY id_apolice
),
base AS (
    SELECT
        a.id_apolice,
        a.regiao,
        a.exposicao,
        a.premio_ganho,
        COALESCE(s.quantidade_sinistros, 0) AS quantidade_sinistros,
        COALESCE(s.custo_sinistros, 0) AS custo_sinistros
    FROM apolices a
    LEFT JOIN sinistros_por_apolice s
        ON a.id_apolice = s.id_apolice
)
SELECT
    regiao,
    COUNT(*) AS apolices,
    SUM(exposicao) AS exposicao,
    SUM(quantidade_sinistros) AS sinistros,
    SUM(premio_ganho) AS premio_ganho,
    SUM(custo_sinistros) AS custo_sinistros,
    SUM(quantidade_sinistros) * 1.0 / SUM(exposicao) AS frequencia,
    SUM(custo_sinistros) * 1.0
        / NULLIF(SUM(quantidade_sinistros), 0) AS severidade,
    SUM(custo_sinistros) * 1.0 / SUM(premio_ganho) AS loss_ratio
FROM base
GROUP BY regiao
ORDER BY regiao;
'''

resumo_regiao = pd.read_sql_query(query, conexao)
print("\nResumo atuarial por região:")
print(resumo_regiao)

# ============================================================
# 15. Criação da VIEW de modelagem
# ============================================================

conexao.execute("DROP VIEW IF EXISTS vw_base_modelagem;")

conexao.execute('''
CREATE VIEW vw_base_modelagem AS

WITH sinistros_por_apolice AS (
    SELECT
        id_apolice,
        COUNT(*) AS quantidade_sinistros,
        SUM(valor_sinistro) AS custo_total_sinistros
    FROM sinistros
    GROUP BY id_apolice
)

SELECT
    a.id_apolice,
    a.exposicao,
    a.idade_condutor,
    a.idade_veiculo,
    a.regiao,
    a.tipo_veiculo,
    a.tipo_uso,
    a.nivel_cobertura,
    a.classe_bonus,
    a.premio_vigente,
    a.premio_ganho,
    COALESCE(s.quantidade_sinistros, 0) AS quantidade_sinistros,
    COALESCE(s.custo_total_sinistros, 0) AS custo_total_sinistros
FROM apolices a
LEFT JOIN sinistros_por_apolice s
    ON a.id_apolice = s.id_apolice;
''')

conexao.commit()

# ============================================================
# 16. Teste da VIEW
# ============================================================

base_view = pd.read_sql_query(
    "SELECT * FROM vw_base_modelagem;",
    conexao
)

print("\nDimensão da VIEW vw_base_modelagem:")
print(base_view.shape)

print("\nPrimeiras linhas da VIEW:")
print(base_view.head())

# ============================================================
# 17. Encerramento
# ============================================================

conexao.close()
print("\nEtapa 2 concluída com sucesso.")