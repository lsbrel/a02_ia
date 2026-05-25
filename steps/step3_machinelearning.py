import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
)

def step3_models(data, random_state=42):
    """Treina os dois algoritmos e devolve modelos, predições e métricas.
    data é o dicionário produzido pela ETAPA 2. O random_state garante reprodutibilidade
    """
    X = data["X"]
    y = data["y"]
    colunas_usadas = data["colunas_usadas"]
    nomes_posicoes = data["nomes_posicoes"]

    print("\nETAPA 3A - Árvore de Decisão (Supervisionado)...")

    # Divisão holdout estratificada. stratify=y garante que cada uma das 4
    # classes apareça na mesma proporção em treino e teste, evitando que
    # uma classe minoritária (goleiros) acabe sumindo de um dos lados.

    # Pega os 19.239 jogadores e separa 80% pra treino e 20% pra teste
    # stratify mantem proporção de jogadores pra evitar enviesamento
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    print(f"  Treino: {len(X_treino)} jogadores  |  Teste: {len(X_teste)} jogadores")

    arvore = DecisionTreeClassifier(
        criterion="entropy", # utilizando ganho de informação baseado em entropia como um algoritmo ID3 faria, mas sklearn usa CART
        max_depth=8,  # anti-overfitting (memorização)
        random_state=random_state, # travando a "seed" do resultado aleatorio pra ter mesmo relatorio smp
    )
    arvore.fit(X_treino, y_treino)  # treinamento ("fase de treinamento")
    y_previsto_arvore = arvore.predict(X_teste)  # fase de utilização/teste

    acuracia_holdout = accuracy_score(y_teste, y_previsto_arvore)
    print(f"  Acurácia (holdout 80/20)  : {acuracia_holdout:.2%}")

    # CV k=10, O modelo é treinado e testado 10 vezes em recortes diferentes a média estabiliza o estimador, o desvio mede sua incerteza.
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    scores_cv = cross_val_score(arvore, X, y, cv=kfold, scoring="accuracy")
    acuracia_cv_media = scores_cv.mean()
    acuracia_cv_desvio = scores_cv.std()
    print(f"  Acurácia (CV k=10)        : " f"{acuracia_cv_media:.2%} ± {acuracia_cv_desvio:.2%}")

    # feature_importances_ é o GANHO DE INFORMAÇÃO acumulado de cada
    # atributo somado por todos os nós da árvore. Quanto maior, mais
    # "discriminativo" o atributo foi para o aprendizado.
    importancia_atributos = pd.Series(arvore.feature_importances_, index=colunas_usadas).sort_values(ascending=False)

    print("\n  Top 5 atributos mais informativos:")
    for atributo, importancia in importancia_atributos.head(5).items():
        print(f"    {atributo:30s} {importancia:.4f}")

    # Matriz de confusão: linhas = classe real, colunas = classe prevista.
    # A diagonal são acertos; fora dela são os erros (e indicam QUAIS pares
    # de classes o modelo confunde, o mais comum foi Meia com Atacante).
    matriz_confusao = confusion_matrix(y_teste, y_previsto_arvore)

    # ══════════════════════════════════════════════════════════════════════
    # 3B - K-MEANS (Não-Supervisionado)
    # ══════════════════════════════════════════════════════════════════════
    print("\nETAPA 3B - K-Means (Não-Supervisionado)...")

    # Antes de fixar K=4, vale a pena ver o "Elbow Method": rodamos
    # K-Means para K=2..8 e medimos (a) inércia - soma das distâncias
    # quadradas dos pontos ao centroide; (b) silhouette - o quão bem
    # separados os clusters estão. O "cotovelo" do gráfico de inércia
    # geralmente indica o K ideal. A ETAPA 4 plota essas curvas.
    lista_inercias = []
    lista_silhouettes = []
    valores_k = list(range(2, 9))
    for k in valores_k:
        km_teste = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_teste.fit(X)
        lista_inercias.append(km_teste.inertia_)
        lista_silhouettes.append(silhouette_score(X, km_teste.labels_))

    # Fixamos K = número real de classes para conseguir comparar
    # clusters = posições. n_init=10 roda o K-Means 10 vezes com
    # inicializações aleatórias diferentes e fica com a melhor -
    # essa é a "dependência de boa inicialização" citada
    # como ponto fraco do K-Means na aula.
    numero_de_posicoes = len(nomes_posicoes) # 4
    kmeans = KMeans(n_clusters=numero_de_posicoes, random_state=random_state, n_init=10)
    clusters_encontrados = kmeans.fit_predict(X)

    silhouette_final = silhouette_score(X, clusters_encontrados)
    ari_final = adjusted_rand_score(y, clusters_encontrados)

    # ── Tradução cluster = posição ────────────────────────────────────────
    # O K-Means não conhece os nomes das classes; ele rotula como 0/1/2/3
    # arbitrariamente. Para comparar a clusterização com os rótulos reais,
    # mapeamos cada cluster para a posição mais frequente dentro dele
    # (voto majoritário). Esse é um truque comum em avaliação externa
    # de clustering.
    mapa_cluster_para_posicao = {}
    for id_cluster in range(numero_de_posicoes):
        rotulos_no_cluster = y[clusters_encontrados == id_cluster]
        if len(rotulos_no_cluster) == 0:
            mapa_cluster_para_posicao[id_cluster] = 0
            continue
        valores, contagens = np.unique(rotulos_no_cluster, return_counts=True)
        mapa_cluster_para_posicao[id_cluster] = valores[np.argmax(contagens)]

    clusters_como_posicoes = np.array(
        [mapa_cluster_para_posicao[c] for c in clusters_encontrados]
    )
    acuracia_kmeans = accuracy_score(y, clusters_como_posicoes)

    print(
        f"  Silhouette Score          : {silhouette_final:.3f}  "
        f"(quanto maior, mais separados os clusters)"
    )
    print(
        f"  Adjusted Rand Index (ARI) : {ari_final:.3f}  "
        f"(quanto maior, mais parecido com a verdade)"
    )
    print(
        f"  Acurácia (vs rótulo real) : {acuracia_kmeans:.2%}  "
        f"(após mapeamento por voto)"
    )

    # Objeto para etapa de comparação
    return {
        # --- Árvore ---
        "arvore": arvore,
        "X_treino": X_treino,
        "X_teste": X_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
        "y_previsto_arvore": y_previsto_arvore,
        "acuracia_holdout": acuracia_holdout,
        "scores_cv": scores_cv,
        "acuracia_cv_media": acuracia_cv_media,
        "acuracia_cv_desvio": acuracia_cv_desvio,
        "importancia_atributos": importancia_atributos,
        "matriz_confusao": matriz_confusao,
        # --- K-Means ---
        "kmeans": kmeans,
        "clusters_encontrados": clusters_encontrados,
        "valores_k": valores_k,
        "lista_inercias": lista_inercias,
        "lista_silhouettes": lista_silhouettes,
        "silhouette_final": silhouette_final,
        "ari_final": ari_final,
        "acuracia_kmeans": acuracia_kmeans,
        "numero_de_posicoes": numero_de_posicoes,
    }


if __name__ == "__main__":
    from step1_load_data import step1_load
    from step2_normalize_data import step2_preprocess

    step3_models(step2_preprocess(step1_load()))
