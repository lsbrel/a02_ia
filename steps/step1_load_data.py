from pathlib import Path
import pandas as pd

# Importando de arquivo CSV
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "database" / "players_22.csv"


def step1_load(csv_path=CSV_PATH):
    print("ETAPA 1 - Carregando o dataset FIFA 22...")

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}\n")

    # remoção de warnings
    tabela_original = pd.read_csv(csv_path, low_memory=False)

    print(f"  Numéro de linhas (jogadores) : {len(tabela_original):>6}")
    print(f"  Númerop de atrivutos (colunas)    : {len(tabela_original.columns):>6}")
    return tabela_original


if __name__ == "__main__":
    step1_load()
