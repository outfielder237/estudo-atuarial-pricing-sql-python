-- Schema used in Stage 2
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS apolices (
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

CREATE TABLE IF NOT EXISTS sinistros (
    id_sinistro INTEGER PRIMARY KEY,
    id_apolice INTEGER NOT NULL,
    valor_sinistro REAL NOT NULL,
    FOREIGN KEY (id_apolice)
        REFERENCES apolices(id_apolice)
);
