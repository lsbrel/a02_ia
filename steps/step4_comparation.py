"""
================================================================================
ETAPA 4 - Analise dos resultados e visualizacoes
================================================================================

   1. Curva de aprendizado
      Como a acuracia da Arvore evolui conforme aumentamos o treino.
      O gap entre treino e teste e o indicador de overfitting.

   2. Matriz de confusao normalizada (%)
      Mostra ONDE a Arvore acerta e erra, em proporcao por classe real.
      Normalizar por linha deixa muito mais facil de ler que contagens.

   3. Estrutura real (PCA + rotulos verdadeiros)
      Projecao 2D dos 12 atributos colorida pelas posicoes REAIS dos
      jogadores. Mostra o ground truth.

   4. Descoberta do K-Means (PCA + clusters mapeados)
      Mesmo grafico, agora colorido pelos clusters do K-Means (mapeados
      para a posicao mais frequente em cada cluster).
================================================================================
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent   # fifa/

def step4_compare(data, models, output_dir=BASE_DIR, show=True):
    """Renderiza o painel 2x2 e o grafico bonus de importancia."""
    print("\nETAPA 4 - Gerando graficos e resumo final...")

    # ── Desempacotamento ──────────────────────────────────────────────────
    X = data["X"]
    y = data["y"]
    nomes_posicoes = data["nomes_posicoes"]
    tabela = data["tabela"]
    colunas_usadas = data["colunas_usadas"]

    acuracia_holdout = models["acuracia_holdout"]
    acuracia_cv_media = models["acuracia_cv_media"]
    acuracia_cv_desvio = models["acuracia_cv_desvio"]
    matriz_confusao = models["matriz_confusao"]
    clusters_encontrados = models["clusters_encontrados"]
    silhouette_final = models["silhouette_final"]
    ari_final = models["ari_final"]
    acuracia_kmeans = models["acuracia_kmeans"]
    importancia_atributos = models["importancia_atributos"]
    numero_de_posicoes = models["numero_de_posicoes"]

    # ── Preparacoes compartilhadas ────────────────────────────────────────
    # PCA: projetamos as 12 dimensoes em 2D uma unica vez, e usamos a mesma
    # projecao nos graficos 3 e 4. Isso garante que estamos comparando
    # exatamente os mesmos pontos, so com cores diferentes.
    redutor_pca = PCA(n_components=2, random_state=42)
    X_em_2d = redutor_pca.fit_transform(X)

    # Mapeamento cluster -> posicao (voto majoritario). Permite usar a
    # MESMA paleta de cores para rotulos reais e clusters do K-Means:
    # se a Posicao X for verde no grafico 3, o cluster que virou X tambem
    # sera verde no grafico 4. Cores iguais = K-Means acertou.
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

    # Paleta fixa: uma cor por posicao (sempre na mesma ordem alfabetica
    # do LabelEncoder). Usar cores categoricas claras facilita a leitura.
    paleta = plt.get_cmap("tab10")
    cores_por_classe = {i: paleta(i) for i in range(len(nomes_posicoes))}

    # ══════════════════════════════════════════════════════════════════════
    # PAINEL 2x2
    # ══════════════════════════════════════════════════════════════════════
    fig, eixos = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("FIFA 22: Arvore de Decisao (supervisionado) vs K-Means (nao supervisionado)", fontsize=15, fontweight="bold")

    # ── Grafico 1: Curva de aprendizado ───────────────────────────────────
    # Treinamos a arvore em frações crescentes dos dados e medimos
    # acuracia em treino e teste. Treino sempre alto; teste sobe e
    # estabiliza. Gap pequeno = boa generalizacao.
    percentuais_treino = [10, 20, 30, 40, 50, 60, 70, 80]
    acuracias_no_treino, acuracias_no_teste = [], []
    for pct in percentuais_treino:
        Xt, Xts, yt, yts = train_test_split(
            X, y, train_size=pct / 100, random_state=42, stratify=y,
        )
        modelo_temp = DecisionTreeClassifier(
            criterion="entropy", max_depth=8, random_state=42,
        )
        modelo_temp.fit(Xt, yt)
        acuracias_no_treino.append(modelo_temp.score(Xt, yt))
        acuracias_no_teste.append(modelo_temp.score(Xts, yts))

    ax1 = eixos[0, 0]
    ax1.plot(percentuais_treino, acuracias_no_treino, "o-",
             color="#4a90e2", label="Acerto no treino", linewidth=2.5,
             markersize=8)
    ax1.plot(percentuais_treino, acuracias_no_teste, "s-",
             color="#e74c3c", label="Acerto em dados novos (teste)",
             linewidth=2.5, markersize=8)
    ax1.fill_between(percentuais_treino,
                     acuracias_no_treino, acuracias_no_teste,
                     alpha=0.12, color="gray", label="Gap (overfitting)")
    ax1.set_title("Como a Arvore aprende\n(treino vs dados nunca vistos)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Quantidade de dados usados no treino (%)")
    ax1.set_ylabel("Acuracia")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_ylim(0.5, 1.02)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(alpha=0.3)

    # ── Grafico 2: Matriz de confusao normalizada por linha ───────────────
    # Normalizar por linha = "para cada classe REAL, em que classes o
    # modelo distribuiu a previsao?". A diagonal vira o recall por
    # classe; um numero perto de 100% significa que o modelo quase
    # nunca confunde aquela classe com nada.
    matriz_pct = (matriz_confusao.astype("float") / matriz_confusao.sum(axis=1, keepdims=True) * 100)
    ax2 = eixos[0, 1]
    sns.heatmap(
        matriz_pct, annot=True, fmt=".1f", ax=ax2, cmap="Greens",
        xticklabels=nomes_posicoes, yticklabels=nomes_posicoes,
        cbar_kws={"label": "% das instancias da classe real"},
        vmin=0, vmax=100, annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax2.set_title("Onde a Árvore acerta e erra\n(% por classe real, diagonal = acerto)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Posição prevista pelo modelo")
    ax2.set_ylabel("Posição real")

    # ── Grafico 3: PCA com rotulos REAIS ──────────────────────────────────
    # Os 12 atributos viraram 2 componentes principais. Cada ponto e um
    # jogador, colorido pela sua posicao verdadeira. Mostra a estrutura
    # natural do dataset.
    ax3 = eixos[1, 0]
    for codigo_classe, nome in enumerate(nomes_posicoes):
        mask = (y == codigo_classe)
        ax3.scatter(
            X_em_2d[mask, 0], X_em_2d[mask, 1],
            c=[cores_por_classe[codigo_classe]],
            alpha=0.45, s=12, label=nome,
        )
    ax3.set_title("Estrutura real do dataset\n(PCA 2D + posições verdadeiras)",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("Eixo 1 - Linha vs Goleiro  (76% var.)")
    ax3.set_ylabel("Eixo 2 - Defensor vs Atacante  (16% var.)")
    legenda3 = ax3.legend(loc="best", framealpha=0.95, markerscale=2.5, title="Posicao real")
    for handle in legenda3.legend_handles:
        handle.set_alpha(1.0)

    # ── Grafico 4: PCA com clusters do K-Means ────────────────────────────
    # Mesmos pontos, mesmas coordenadas, mesmas cores. Mas a cor agora
    # vem do CLUSTER que o K-Means descobriu sem ver os rotulos.
    # Cor diferente da do grafico 3 = K-Means classificou diferente
    # do humano.
    ax4 = eixos[1, 1]
    for codigo_classe, nome in enumerate(nomes_posicoes):
        mask = (clusters_como_posicoes == codigo_classe)
        ax4.scatter(
            X_em_2d[mask, 0], X_em_2d[mask, 1],
            c=[cores_por_classe[codigo_classe]],
            alpha=0.45, s=12, label=nome,
        )
    ax4.set_title(
        f"Descoberta do K-Means sozinho\n"
        f"(Acuracia vs real: {acuracia_kmeans:.1%}  |  ARI: {ari_final:.2f})",
        fontsize=12, fontweight="bold",
    )
    ax4.set_xlabel("Eixo 1 - Linha vs Goleiro  (76% var.)")
    ax4.set_ylabel("Eixo 2 - Defensor vs Atacante  (16% var.)")
    legenda4 = ax4.legend(loc="best", framealpha=0.95, markerscale=2.5, title="Cluster (após mapeamento)")
    for handle in legenda4.legend_handles:
        handle.set_alpha(1.0)

    plt.tight_layout()
    caminho_painel = output_dir / "resultados_fifa.png"
    plt.savefig(caminho_painel, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  -> painel salvo em: {caminho_painel}")

    # ── Grafico final: importancia dos atributos ──────────────────────────
    # Resposta para "o que o modelo realmente aprendeu". Atributos com
    # maior Ganho de Informação acumulado são os que mais discriminam entre as 4 classes.
    fig2, eixo_imp = plt.subplots(figsize=(10, 6))
    top5_threshold = importancia_atributos.head(5).min()
    importancia_ord = importancia_atributos.sort_values()
    cores_imp = [
        "#27ae60" if v >= top5_threshold else "#95a5a6"
        for v in importancia_ord.values
    ]
    importancia_ord.plot(
        kind="barh", ax=eixo_imp, color=cores_imp, edgecolor="white",
    )
    for i, valor in enumerate(importancia_ord.values):
        eixo_imp.text(valor + 0.005, i, f"{valor:.1%}",
                      va="center", fontsize=9)
    eixo_imp.set_title(
        "O que a Árvore considerou mais importante\n"
        "(verde = top 5 atributos com maior Ganho de Informação)",
        fontsize=13, fontweight="bold",
    )
    eixo_imp.set_xlabel("Importancia (% do Ganho de Informação total)")
    eixo_imp.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    eixo_imp.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    caminho_imp = output_dir / "importancia_atributos_fifa.png"
    plt.savefig(caminho_imp, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)
    print(f"  -> importancia salva em: {caminho_imp}")

    # ── Resumo final em texto ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RESUMO FINAL")
    print("=" * 64)
    print(f"  Dataset       : FIFA 22 - {len(tabela)} jogadores, "
          f"{len(colunas_usadas)} atributos")
    print(f"  Classes alvo  : {list(nomes_posicoes)}")
    print()
    print("  ARVORE DE DECISAO (supervisionada)")
    print(f"    Holdout 80/20      : {acuracia_holdout:.2%}")
    print(f"    CV estratificada   : {acuracia_cv_media:.2%} "
          f"+/- {acuracia_cv_desvio:.2%}")
    print()
    print("  K-MEANS (não supervisionado)")
    print(f"    Silhouette Score   : {silhouette_final:.3f}")
    print(f"    Adjusted Rand Idx  : {ari_final:.3f}")
    print(f"    Acuracia (vs real) : {acuracia_kmeans:.2%}")
    print()
    print(f"  Atributo mais informativo: {importancia_atributos.index[0]} "
          f"({importancia_atributos.iloc[0]:.1%})")
    print("=" * 64)


if __name__ == "__main__":
    from step1_load_data import step1_load
    from step2_normalize_data import step2_preprocess
    from step3_machinelearning import step3_models
    data = step2_preprocess(step1_load())
    models = step3_models(data)
    step4_compare(data, models)
