import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

COLUNAS_USADAS = [
    "pace",  # velocidade - NaN para goleiros (FIFA so calcula para jogadores de linha)
    "shooting",  # chute - NaN para goleiros
    "passing",  # passe - NaN para goleiros
    "dribbling",  # drible - NaN para goleiros
    "defending",  # marcacao geral - NaN para goleiros
    "physic",  # forca fisica - NaN para goleiros
    "goalkeeping_diving",  # mergulho - existe para todos, alto so em goleiros
    "goalkeeping_handling",  # pegar a bola
    "goalkeeping_reflexes",  # reflexos
    "attacking_finishing",  # finalizacao
    "defending_standing_tackle",  # carrinho parado
    "mentality_positioning",  # posicionamento em campo
]

# Subconjunto das colunas acima que sao NaN para goleiros no dataset original.
# Trataremos esses NaN como 0 (semanticamente: o FIFA nao mede essas habilidades
# em goleiros, e o valor zero deixa a classe Goleiro trivialmente separavel).
COLUNAS_SO_DE_LINHA = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
]


def agrupar_posicao(posicao_raw):
    """Converte a posição original do FIFA em uma das 4 categorias agregadas."""
    if pd.isna(posicao_raw):
        return None

    # Jogadores com mais de uma posição utilizamos apenas a primeira
    pos = str(posicao_raw).split(",")[0].strip().upper()

    if pos in {"GK"}:
        return "Goleiro"
    if pos in {"CB", "LB", "RB", "LWB", "RWB"}:
        return "Defensor"
    if pos in {"CM", "CAM", "CDM", "LM", "RM", "LW", "RW"}:
        return "Meia"
    if pos in {"ST", "CF", "LF", "RF", "LS", "RS", "SS"}:
        return "Atacante"
    return None  # Caso tenha alguma posição não mapeada descartamo


def step2_preprocess(tabela_original):
    """Aplica toda a sequência de pré-processamento descrita no docstring acima.

    Recebe o DataFrame bruto da Etapa 1 e devolve um dicionário contendo:
      - X              : matriz de atributos já normalizada (NumPy);
      - y              : rótulos codificados em inteiros;
      - tabela         : DataFrame limpo (útil para inspeção e gráficos);
      - colunas_usadas : lista dos atributos, na mesma ordem das colunas de X;
      - nomes_posicoes : nomes legíveis das classes (na ordem dos códigos);
      - scaler/encoder : objetos sklearn ajustados (caso queiramos transformar
                         novos jogadores depois sem refazer a estatística).
    """
    print("\nETAPA 2 - Pré-processamento...")

    tabela_original = tabela_original.copy()
    tabela_original["posicao_simples"] = tabela_original["player_positions"].apply(
        agrupar_posicao
    )

    # Mantemos apenas o subconjunto relevante (atributos + alvo).
    tabela = tabela_original[COLUNAS_USADAS + ["posicao_simples"]].copy()

    # IMPORTANTE: os goleiros do FIFA tem NaN nas 6 estatisticas de jogadores
    # de linha (pace, shooting, passing, dribbling, defending, physic), porque
    # o jogo nao calcula esses ratings para a posicao. Se simplesmente
    # chamassemos dropna() aqui, perderiamos TODOS os goleiros (2132 instancias)
    # e ficariamos com apenas 3 classes. Preenchemos esses NaN com 0: zero e
    # interpretavel como "o FIFA nao mede essa habilidade neste jogador" e da
    # ao classificador uma assinatura ultra-clara para detectar goleiros.
    tabela[COLUNAS_SO_DE_LINHA] = tabela[COLUNAS_SO_DE_LINHA].fillna(0)

    # Agora o dropna serve apenas para remover jogadores com posicao nao
    # mapeada (raras combinacoes que cairam no return None de agrupar_posicao).
    tabela = tabela.dropna(subset=["posicao_simples"]).reset_index(drop=True)

    print(f"  Instâncias após a limpeza : {len(tabela)}")
    print("  Distribuição por classe   :")
    for posicao, contagem in tabela["posicao_simples"].value_counts().items():
        pct = contagem / len(tabela) * 100
        print(f"    {posicao:10s} {contagem:>6}  ({pct:5.2f}%)")

    # Segunda camada de defesa: se ainda restou algum NaN, imputamos com a
    # mediana da coluna. Mediana > média porque é mais robusta a outliers
    # (jogadores estrelas com ratings 95+).
    qtd_faltantes = tabela[COLUNAS_USADAS].isnull().sum().sum()
    if qtd_faltantes > 0:
        for coluna in COLUNAS_USADAS:
            tabela[coluna] = tabela[coluna].fillna(tabela[coluna].median())
        print(f"  {qtd_faltantes} valores faltantes preenchidos com a mediana.")
    else:
        print("  Nenhum valor faltante remanescente.")

    # sklearn só aceita y numérico. LabelEncoder é o mais simples: rotula em
    # ordem alfabética. Para classes ordinais usaríamos OrdinalEncoder com
    # ordem explícita; para nominais com cardinalidade alta, OneHotEncoder.
    X_texto = tabela[COLUNAS_USADAS]
    y_texto = tabela["posicao_simples"]

    codificador_posicao = LabelEncoder()
    y = codificador_posicao.fit_transform(y_texto)
    nomes_posicoes = codificador_posicao.classes_
    print(
        f"\n  Codificação: {list(nomes_posicoes)} "
        f"-> {list(range(len(nomes_posicoes)))}"
    )

    # X_norm = (X - média) / desvio_padrão, coluna a coluna. Indispensável
    # para o K-Means; inofensivo para a Árvore.
    normalizador = StandardScaler()
    X = normalizador.fit_transform(X_texto)
    print(f"  Matriz X final: {X.shape[0]} linhas x {X.shape[1]} atributos.")

    return {
        "X": X,
        "y": y,
        "tabela": tabela,
        "colunas_usadas": COLUNAS_USADAS,
        "nomes_posicoes": nomes_posicoes,
        "scaler": normalizador,
        "label_encoder": codificador_posicao,
    }


if __name__ == "__main__":
    from step1_load_data import step1_load

    step2_preprocess(step1_load())
