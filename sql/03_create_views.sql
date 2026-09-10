DROP VIEW IF EXISTS vw_base_modelagem;

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
