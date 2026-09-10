CREATE INDEX IF NOT EXISTS idx_sinistros_id_apolice
ON sinistros(id_apolice);

CREATE INDEX IF NOT EXISTS idx_apolices_regiao
ON apolices(regiao);

CREATE INDEX IF NOT EXISTS idx_apolices_tipo_veiculo
ON apolices(tipo_veiculo);
