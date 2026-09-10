-- Basic integrity and reconciliation checks

SELECT COUNT(*) AS quantidade_apolices
FROM apolices;

SELECT COUNT(*) AS quantidade_sinistros
FROM sinistros;

SELECT COUNT(*) AS sinistros_orfaos
FROM sinistros s
LEFT JOIN apolices a
    ON s.id_apolice = a.id_apolice
WHERE a.id_apolice IS NULL;

SELECT
    SUM(a.exposicao) AS exposicao_total,
    SUM(a.premio_ganho) AS premio_ganho_total,
    SUM(COALESCE(v.custo_total_sinistros, 0)) AS custo_sinistros_total
FROM apolices a
LEFT JOIN vw_base_modelagem v
    ON a.id_apolice = v.id_apolice;
