
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def features_gd(df, target_value):
    """
        Tomara los conjuntos que genera cada una de las clases del dataset
        para cada una de las features y mostrara la distribucion gaussiana de cada una.

        El objetivo es revisar si la variacion de clases conlleva una distribucion de 
        valores.

        Soporta hasta 2 clases.
    """
    features = df.drop(target_value, axis=1)

    plt.figure(figsize=(16, 32))
    gs = gridspec.GridSpec(8, 4)
    gs.update(hspace=0.8)

    for i, f in enumerate(features):
        ax = plt.subplot(gs[i])
        sns.histplot(data=df[df[target_value] == 1], x=f, kde=True, color="red", stat="density", label="x = 1", alpha=0.5)
        sns.histplot(data=df[df[target_value] == 0], x=f, kde=True, color="green", stat="density", label="x = 0", alpha=0.5)
        ax.set_xlabel('')
        ax.set_title(f"Feature: {f}")
        ax.legend()

    plt.show()
