from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Os PNGs são salvos em fifa/ (uma pasta acima de steps/) para ficarem
# junto do README/relatório, e não soltos dentro de steps/.
BASE_DIR = Path(__file__).resolve().parent.parent  # → fifa/


def step4_compare(data, models, output_dir=BASE_DIR, show=True):
    """Renderiza o painel 2×3 e o gráfico bônus de importância. Salva PNGs."""
    print("\nETAPA 4 - Gerando gráficos e resumo final...")

    # ── Desempacotamento para deixar o código de plot mais legível ────────
    X = data["X"]
    y = data["y"]
    nomes_posicoes = data["nomes_posicoes"]
    tabela = data["tabela"]
    colunas_usadas = data["colunas_usadas"]

    acuracia_holdout = models["acuracia_holdout"]
    acuracia_cv_media = models["acuracia_cv_media"]
    acuracia_cv_desvio = models["acuracia_cv_desvio"]
    scores_cv = models["scores_cv"]
    matriz_confusao = models["matriz_confusao"]
    valores_k = models["valores_k"]
    lista_inercias = models["lista_inercias"]
    lista_silhouettes = models["lista_silhouettes"]
    clusters_encontrados = models["clusters_encontrados"]
    silhouette_final = models["silhouette_final"]
    ari_final = models["ari_final"]
    acuracia_kmeans = models["acuracia_kmeans"]
    importancia_atributos = models["importancia_atributos"]
    numero_de_posicoes = models["numero_de_posicoes"]

    fig, eixos = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        "Resultados - FIFA 22: Árvore de Decisão vs K-Means",
        fontsize=15,
        fontweight="bold",
    )

    # ── Gráfico 1: Curva de aprendizado (memorização vs generalização) ───
    # Treinamos a árvore em frações progressivas dos dados e medimos a
    # acurácia tanto nos dados de treino quanto nos de teste. O gap entre
    # as duas curvas é exatamente o indicador de overfitting visto na aula.
    percentuais_treino = [10, 20, 30, 40, 50, 60, 70, 80]
    acuracias_no_treino, acuracias_no_teste = [], []
    for pct in percentuais_treino:
        Xt, Xts, yt, yts = train_test_split(
            X,
            y,
            train_size=pct / 100,
            random_state=42,
            stratify=y,
        )
        modelo_temp = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=8,
            random_state=42,
        )
        modelo_temp.fit(Xt, yt)
        acuracias_no_treino.append(modelo_temp.score(Xt, yt))
        acuracias_no_teste.append(modelo_temp.score(Xts, yts))

    eixos[0, 0].plot(
        percentuais_treino,
        acuracias_no_treino,
        "o-",
        color="steelblue",
        label="Treino",
        linewidth=2,
    )
    eixos[0, 0].plot(
        percentuais_treino,
        acuracias_no_teste,
        "s-",
        color="tomato",
        label="Teste",
        linewidth=2,
    )
    eixos[0, 0].set_title("Curva de Aprendizado (Árvore)")
    eixos[0, 0].set_xlabel("% dos dados usados no treino")
    eixos[0, 0].set_ylabel("Acurácia")
    eixos[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    eixos[0, 0].legend()
    eixos[0, 0].grid(alpha=0.3)

    # ── Gráfico 2: Validação cruzada k=10 fold a fold ────────────────────
    # Cor verde = fold acima da média, vermelho = abaixo. Folds com
    # variação pequena entre si indicam que o modelo é estável.
    cores_fold = ["seagreen" if s >= acuracia_cv_media else "tomato" for s in scores_cv]
    eixos[0, 1].bar(range(1, 11), scores_cv, color=cores_fold, edgecolor="white")
    eixos[0, 1].axhline(
        acuracia_cv_media,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Média: {acuracia_cv_media:.1%}",
    )
    eixos[0, 1].set_title("Validação Cruzada k=10 (Árvore)")
    eixos[0, 1].set_xlabel("Fold")
    eixos[0, 1].set_ylabel("Acurácia")
    eixos[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    eixos[0, 1].legend()
    eixos[0, 1].grid(axis="y", alpha=0.3)

    # ── Gráfico 3: Matriz de confusão (mostra ONDE o modelo erra) ────────
    # Diagonal = acertos. Fora da diagonal = erros, com a identidade do
    # par confundido. Tipicamente Meia ↔ Atacante são os mais misturados.
    sns.heatmap(
        matriz_confusao,
        annot=True,
        fmt="d",
        ax=eixos[0, 2],
        cmap="Blues",
        xticklabels=nomes_posicoes,
        yticklabels=nomes_posicoes,
    )
    eixos[0, 2].set_title("Matriz de Confusão (Árvore - teste)")
    eixos[0, 2].set_xlabel("Posição prevista")
    eixos[0, 2].set_ylabel("Posição real")

    # ── Gráfico 4: Elbow Method (escolha do K) ───────────────────────────
    # Duas curvas no mesmo eixo X com escalas diferentes nos dois eixos Y.
    # A linha vertical pontilhada marca o K que escolhemos (= 4 classes).
    eixo_esq = eixos[1, 0]
    eixo_dir = eixo_esq.twinx()
    eixo_esq.plot(valores_k, lista_inercias, "o-", color="steelblue", label="Inércia")
    eixo_dir.plot(
        valores_k, lista_silhouettes, "s--", color="darkorange", label="Silhouette"
    )
    eixo_esq.axvline(
        numero_de_posicoes,
        color="gray",
        linestyle=":",
        label=f"K escolhido = {numero_de_posicoes}",
    )
    eixo_esq.set_title("Elbow Method - escolha de K")
    eixo_esq.set_xlabel("Número de Clusters (K)")
    eixo_esq.set_ylabel("Inércia", color="steelblue")
    eixo_dir.set_ylabel("Silhouette Score", color="darkorange")
    linhas_esq, labels_esq = eixo_esq.get_legend_handles_labels()
    linhas_dir, labels_dir = eixo_dir.get_legend_handles_labels()
    eixo_esq.legend(linhas_esq + linhas_dir, labels_esq + labels_dir, fontsize=8)
    eixo_esq.grid(alpha=0.3)

    # ── Gráfico 5: Clusters em 2D via PCA ────────────────────────────────
    # Não conseguimos visualizar 12 dimensões; o PCA encontra as 2
    # combinações lineares dos atributos que MAIS preservam variância e
    # projeta cada jogador num ponto (x, y). Se os clusters aparecem como
    # ilhas coloridas, o K-Means achou estrutura real.
    redutor_pca = PCA(n_components=2, random_state=42)
    X_em_2d = redutor_pca.fit_transform(X)
    eixos[1, 1].scatter(
        X_em_2d[:, 0],
        X_em_2d[:, 1],
        c=clusters_encontrados,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    eixos[1, 1].set_title(
        f"Clusters K-Means projetados em 2D (PCA)\n"
        f"Silhouette={silhouette_final:.3f} | ARI={ari_final:.3f}"
    )
    eixos[1, 1].set_xlabel("Componente Principal 1")
    eixos[1, 1].set_ylabel("Componente Principal 2")

    # ── Gráfico 6: Comparativo Árvore vs K-Means ─────────────────────────
    # K-Means não tem CV (não faz sentido - ele não treina/prevê), então
    # repetimos seu valor de acurácia nas duas colunas só para alinhar
    # visualmente as barras.
    nomes_metricas = ["Holdout\n(80/20)", "CV k=10\n(média)"]
    valores_arvore = [acuracia_holdout, acuracia_cv_media]
    valores_kmeans = [acuracia_kmeans, acuracia_kmeans]
    posicoes_barras = np.arange(len(nomes_metricas))
    largura = 0.3

    eixos[1, 2].bar(
        posicoes_barras - largura / 2,
        valores_arvore,
        largura,
        label="Árvore de Decisão",
        color="steelblue",
        alpha=0.85,
    )
    eixos[1, 2].bar(
        posicoes_barras + largura / 2,
        valores_kmeans,
        largura,
        label="K-Means",
        color="darkorange",
        alpha=0.85,
    )
    for i, (va, vk) in enumerate(zip(valores_arvore, valores_kmeans)):
        eixos[1, 2].text(
            i - largura / 2, va + 0.005, f"{va:.1%}", ha="center", fontweight="bold"
        )
        eixos[1, 2].text(
            i + largura / 2, vk + 0.005, f"{vk:.1%}", ha="center", fontweight="bold"
        )
    eixos[1, 2].set_title("Comparativo Final")
    eixos[1, 2].set_xticks(posicoes_barras)
    eixos[1, 2].set_xticklabels(nomes_metricas)
    eixos[1, 2].set_ylabel("Acurácia")
    eixos[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    eixos[1, 2].legend()
    eixos[1, 2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    caminho_painel = output_dir / "resultados_fifa.png"
    plt.savefig(caminho_painel, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  → painel salvo em: {caminho_painel}")

    # ── Gráfico bônus: importância dos atributos (conhecimento aprendido) ─
    # Esse é o equivalente acadêmico de "abrir a caixa-preta": ver QUAL
    # atributo a árvore considerou mais relevante para diferenciar as
    # 4 classes. Pista esperada: os atributos de goleiro devem dominar,
    # porque separar goleiros dos demais é trivial.
    fig2, eixo_imp = plt.subplots(figsize=(9, 6))
    top5_threshold = importancia_atributos.head(5).min()
    importancia_ord = importancia_atributos.sort_values()
    cores_imp = [
        "seagreen" if v >= top5_threshold else "steelblue"
        for v in importancia_ord.values
    ]
    importancia_ord.plot(
        kind="barh",
        ax=eixo_imp,
        color=cores_imp,
        edgecolor="white",
        alpha=0.85,
    )
    eixo_imp.set_title(
        "Importância dos Atributos - Árvore de Decisão\n"
        "(verde = top 5 com maior Ganho de Informação)",
        fontsize=13,
    )
    eixo_imp.set_xlabel("Importância acumulada (Ganho de Informação)")
    eixo_imp.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    caminho_imp = output_dir / "importancia_atributos_fifa.png"
    plt.savefig(caminho_imp, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)
    print(f"  → importância salva em: {caminho_imp}")

    # ── Resumo final em texto ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RESUMO FINAL")
    print("=" * 64)
    print(
        f"  Dataset       : FIFA 22 - {len(tabela)} jogadores, "
        f"{len(colunas_usadas)} atributos"
    )
    print(f"  Classes alvo  : {list(nomes_posicoes)}")
    print()
    print("  ÁRVORE DE DECISÃO (supervisionada)")
    print(f"    Holdout 80/20      : {acuracia_holdout:.2%}")
    print(
        f"    CV estratificada   : {acuracia_cv_media:.2%} "
        f"± {acuracia_cv_desvio:.2%}"
    )
    print()
    print("  K-MEANS (não-supervisionado)")
    print(f"    Silhouette Score   : {silhouette_final:.3f}")
    print(f"    Adjusted Rand Idx  : {ari_final:.3f}")
    print(f"    Acurácia (vs real) : {acuracia_kmeans:.2%}")
    print()
    print(f"  Atributo mais informativo: {importancia_atributos.index[0]}")
    print("=" * 64)


if __name__ == "__main__":
    from step1_load_data import step1_load
    from step2_normalize_data import step2_preprocess
    from step3_machinelearning import step3_models

    data = step2_preprocess(step1_load())
    models = step3_models(data)
    step4_compare(data, models)
