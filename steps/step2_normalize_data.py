"""
================================================================================
ETAPA 2 - Pré-processamento dos dados
================================================================================

Como a aula deixou claro, "técnicas tradicionais não são adequadas" quando os
dados estão brutos: temos colunas demais, escalas diferentes, valores faltantes
e classes de saída granulares demais. Esta etapa cuida disso seguindo a
sequência clássica do KDD:

  1. SELEÇÃO DE ATRIBUTOS
       Escolhemos 12 colunas relevantes para distinguir as posições. Atributos
       como nome, foto ou nacionalidade são removidos porque seriam ruído ou,
       pior, identificadores irrelevantes (slide "Características irrelevantes
       - ex: ID do estudante para prever desempenho").

  2. CRIAÇÃO/AGREGAÇÃO DE CLASSES
       O FIFA tem ~30 posições (ST, CF, CAM, CM, CDM, CB, LB, RB, GK...). Tratar
       todas separadamente daria classes muito desbalanceadas (raras como SS ou
       LF têm pouquíssimos exemplos) e o modelo memorizaria em vez de
       generalizar. Agregamos em 4 grandes papéis táticos: Goleiro, Defensor,
       Meia e Atacante. Isso é exatamente o conceito de AGREGAÇÃO da aula
       ("combinar dois ou mais atributos ou objetos em um único") aplicado ao
       rótulo.

  3. LIMPEZA (DADOS FALTANTES)
       A coluna 'pace' não existe para goleiros no FIFA. Após o dropna
       defensivo, ainda preenchemos com a MEDIANA por coluna - opção robusta a
       outliers, citada no slide "Valores Faltantes - estimar valores
       faltantes".

  4. CODIFICAÇÃO DA CLASSE
       O scikit-learn exige y numérico. LabelEncoder mapeia
       'Atacante'→0, 'Defensor'→1, 'Goleiro'→2, 'Meia'→3 (ordem alfabética).

  5. TRANSFORMAÇÃO DE ATRIBUTOS (NORMALIZAÇÃO)
       Aplicamos padronização Z-score: cada coluna passa a ter média 0 e
       desvio 1. Isso é crítico para o K-Means, que se baseia em distância
       euclidiana - sem padronizar, um atributo com amplitude maior dominaria
       o cálculo. Para a Árvore de Decisão isso não muda nada (ela usa
       comparações de limiar, é invariante a escala), mas mantemos um pipeline
       único para os dois algoritmos por simplicidade.
================================================================================
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ──────────────────────────────────────────────────────────────────────────────
# 2.1  ATRIBUTOS SELECIONADOS
# ──────────────────────────────────────────────────────────────────────────────
# A escolha não é arbitrária: pegamos atributos que CONHECIDAMENTE diferenciam
# os papéis táticos no futebol. Esperamos que:
#   • Goleiros se destaquem nas três colunas 'goalkeeping_*'.
#   • Defensores brilhem em 'defending' e 'defending_standing_tackle'.
#   • Atacantes dominem 'shooting', 'attacking_finishing' e 'pace'.
#   • Meias fiquem mais equilibrados em 'passing' e 'dribbling'.
# Essa hipótese será confirmada (ou refutada) pelo gráfico de importância
# dos atributos gerado na ETAPA 4.
COLUNAS_USADAS = [
    'pace',                       # velocidade - diferencia laterais/atacantes
    'shooting',                   # chute - separa atacantes do resto
    'passing',                    # passe - característica de meio-campistas
    'dribbling',                  # drible - atacantes e meias criativos
    'defending',                  # marcação geral - zagueiros e volantes
    'physic',                     # força física - zagueiros e atacantes de área
    'goalkeeping_diving',         # mergulho - exclusivo de goleiros
    'goalkeeping_handling',       # pegar a bola - exclusivo de goleiros
    'goalkeeping_reflexes',       # reflexos - exclusivo de goleiros
    'attacking_finishing',        # finalização - refina a classe Atacante
    'defending_standing_tackle',  # carrinho parado - refina a classe Defensor
    'mentality_positioning',      # posicionamento - útil para todas as classes
]


def agrupar_posicao(posicao_raw):
    """Converte a posição original do FIFA em uma das 4 categorias agregadas."""
    if pd.isna(posicao_raw):
        return None

    # Jogadores com mais de uma posição utilizamos apenas a primeira
    pos = str(posicao_raw).split(',')[0].strip().upper()

    if pos == 'GK':
        return 'Goleiro'
    if pos in {'CB', 'LB', 'RB', 'LWB', 'RWB'}:
        return 'Defensor'
    if pos in {'CM', 'CAM', 'CDM', 'LM', 'RM', 'LW', 'RW'}:
        return 'Meia'
    if pos in {'ST', 'CF', 'LF', 'RF', 'LS', 'RS', 'SS'}:
        return 'Atacante'
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

    # Trabalhamos em uma cópia para não alterar o DataFrame original - boa
    # prática quando outros passos do pipeline ainda podem precisar dele.
    tabela_original = tabela_original.copy()
    tabela_original['posicao_simples'] = (
        tabela_original['player_positions'].apply(agrupar_posicao)
    )

    # Mantemos apenas o subconjunto relevante (atributos + alvo) e descartamos
    # linhas com qualquer campo vazio. Como goleiros têm 'pace'/'shooting'
    # ausentes para jogadores de linha (e vice-versa para os 'goalkeeping_*'),
    # o dropna remove justamente os jogadores cujos atributos não fazem sentido
    # para a posição - não é perda de informação, é coerência.
    tabela = tabela_original[COLUNAS_USADAS + ['posicao_simples']].copy()
    tabela = tabela.dropna().reset_index(drop=True)

    print(f"  Instâncias após a limpeza : {len(tabela)}")
    print("  Distribuição por classe   :")
    for posicao, contagem in tabela['posicao_simples'].value_counts().items():
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

    # ── Codificação da classe alvo ────────────────────────────────────────
    # sklearn só aceita y numérico. LabelEncoder é o mais simples: rotula em
    # ordem alfabética. Para classes ordinais usaríamos OrdinalEncoder com
    # ordem explícita; para nominais com cardinalidade alta, OneHotEncoder.
    X_texto = tabela[COLUNAS_USADAS]
    y_texto = tabela['posicao_simples']

    codificador_posicao = LabelEncoder()
    y = codificador_posicao.fit_transform(y_texto)
    nomes_posicoes = codificador_posicao.classes_
    print(
        f"\n  Codificação: {list(nomes_posicoes)} "
        f"→ {list(range(len(nomes_posicoes)))}"
    )

    # ── Padronização (Z-score) ────────────────────────────────────────────
    # X_norm = (X - média) / desvio_padrão, coluna a coluna. Indispensável
    # para o K-Means; inofensivo para a Árvore.
    normalizador = StandardScaler()
    X = normalizador.fit_transform(X_texto)
    print(f"  Matriz X final: {X.shape[0]} linhas × {X.shape[1]} atributos.")

    return {
        'X': X,
        'y': y,
        'tabela': tabela,
        'colunas_usadas': COLUNAS_USADAS,
        'nomes_posicoes': nomes_posicoes,
        'scaler': normalizador,
        'label_encoder': codificador_posicao,
    }


if __name__ == '__main__':
    from step1_load_data import step1_load
    step2_preprocess(step1_load())
