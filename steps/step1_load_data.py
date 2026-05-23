"""
================================================================================
ETAPA 1 — Definição da base de dados
================================================================================

Trabalhamos com o dataset FIFA 22 (Kaggle), que contém o perfil completo de
todos os jogadores presentes no jogo: nacionalidade, clube, posição em campo
e dezenas de atributos numéricos (de 1 a 99) avaliando habilidades técnicas,
físicas e específicas de goleiro.

Mapeamento ao vocabulário da disciplina:
  - cada linha do CSV → uma INSTÂNCIA (jogador);
  - cada coluna numérica → um ATRIBUTO;
  - 'player_positions'   → o CONCEITO ALVO (o que queremos prever).

Esta etapa apenas carrega o arquivo bruto; toda limpeza acontece na ETAPA 2.

Fonte:
  https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
Localização esperada do CSV:
  fifa/database/players_22.csv
================================================================================
"""
from pathlib import Path

import pandas as pd


# Import do BD
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / 'database' / 'players_22.csv'


def step1_load(csv_path=CSV_PATH):
    """Carrega o CSV original e devolve um DataFrame pandas.
    """
    print("ETAPA 1 — Carregando o dataset FIFA 22...")

    if not csv_path.exists():
        # Caso n encontre o DB
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}\n")

    # low_memory=False evita o aviso do pandas sobre tipos mistos por coluna
    tabela_original = pd.read_csv(csv_path, low_memory=False)

    print(f"  Instâncias (jogadores) : {len(tabela_original):>6}")
    print(f"  Atributos (colunas)    : {len(tabela_original.columns):>6}")
    return tabela_original


if __name__ == '__main__':
    # testar só a leitura do DB
    step1_load()
