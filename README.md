# Pricing & Economic Capital with Python and SQL

Projeto atuarial didático de ponta a ponta para conectar **pricing**, comportamento do cliente,
simulação de perdas e **capital econômico** em uma carteira sintética de seguro Automóvel.

> **O modelo estima o risco. A estratégia decide como correr.**

**Risk & Roots** — *Mergulhar na experiência para discernir a incerteza futura.*

## Pergunta de negócio

**Qual estratégia de pricing oferece o melhor equilíbrio entre crescimento, margem e consumo de capital?**

## Fluxo do projeto

```text
SQL / Dados
    ↓
EDA
    ↓
GLM Frequência + GLM Severidade
    ↓
Prêmio Puro
    ↓
Prêmio Comercial
    ↓
Retenção / Elasticidade
    ↓
Estratégias de Pricing
    ↓
Monte Carlo
    ↓
VaR / TVaR
    ↓
Capital Econômico
    ↓
RoC / EVA / Apetite a Risco
    ↓
Decisão
```

## Etapas

1. Geração da carteira sintética
2. Banco SQLite
3. Análise exploratória
4. GLM de frequência
5. GLM de severidade
6. Prêmio puro
7. Prêmio comercial
8. Retenção e elasticidade
9. Estratégias de pricing
10. Simulação da carteira
11. Capital econômico
12. Retorno sobre capital e decisão final

Os scripts estão em `analysis/` e foram numerados para permitir execução sequencial.

## Estrutura

```text
pricing-capital-python-sql/
├── README.md
├── requirements.txt
├── .gitignore
├── analysis/
├── data/
├── sql/
├── outputs/
│   ├── figuras/
│   └── tabelas/
└── docs/
```

## Como executar

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Depois execute os scripts em ordem, a partir de qualquer diretório:

```bash
python analysis/01_geracao_base_sintetica.py
python analysis/02_banco_sqlite.py
...
python analysis/12_retorno_capital_decisao_final.py
```

Os scripts usam caminhos ancorados na própria raiz do repositório.

## Dados

Todos os dados são **inteiramente sintéticos** e foram criados exclusivamente para fins
educacionais. Nenhum parâmetro deve ser interpretado como referência de mercado, tarifária,
regulatória ou de qualquer seguradora.

## Outputs

A pasta `outputs/` contém tabelas e figuras geradas pelas análises. Arquivos intermediários
muito grandes e reproduzíveis são ignorados pelo Git por padrão.

## Documentação

A pasta `docs/` contém a estrutura para:
- metodologia;
- dicionário de fórmulas;
- dicionário de dados;
- premissas e limitações.

## Licença

Nenhuma licença foi escolhida automaticamente. Defina a licença desejada antes da publicação pública.
