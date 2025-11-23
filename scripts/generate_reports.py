import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def ensure_dirs():
    Path('reports/figs').mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()
    df = pd.read_csv('dataset_alquiler.csv', low_memory=False)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    # Serie temporal: total_alquileres diario
    df_day = df.set_index('fecha').resample('D')['total_alquileres'].sum()
    plt.figure(figsize=(12, 4))
    df_day.plot()
    plt.title('Total alquileres diarios')
    plt.ylabel('total_alquileres')
    plt.tight_layout()
    plt.savefig('reports/figs/series_diaria.png')
    plt.close()

    # Promedio por hora (0-23)
    if 'hora' in df.columns:
        try:
            df['hora'] = df['hora'].astype(float)
            hourly = df.groupby('hora')['total_alquileres'].mean()
            plt.figure(figsize=(8, 4))
            sns.lineplot(x=hourly.index, y=hourly.values)
            plt.title('Promedio de alquileres por hora')
            plt.xlabel('hora')
            plt.ylabel('avg total_alquileres')
            plt.tight_layout()
            plt.savefig('reports/figs/avg_by_hour.png')
            plt.close()
        except Exception:
            pass

    # Histograma total_alquileres
    plt.figure(figsize=(8, 4))
    sns.histplot(df['total_alquileres'].dropna(), bins=50, kde=False)
    plt.xlim(0, df['total_alquileres'].quantile(0.99))
    plt.title('Histograma de total_alquileres (hasta p99)')
    plt.tight_layout()
    plt.savefig('reports/figs/hist_total_alquileres.png')
    plt.close()

    # Correlation heatmap (numéricas)
    num = df.select_dtypes(include=['number']).fillna(0)
    corr = num.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Mapa de correlación (numéricas)')
    plt.tight_layout()
    plt.savefig('reports/figs/correlation_heatmap.png')
    plt.close()

    # Boxplots de sensacion_termica y velocidad_viento
    for col in ['sensacion_termica', 'velocidad_viento']:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot {col}')
            plt.tight_layout()
            plt.savefig(f'reports/figs/box_{col}.png')
            plt.close()

    print('Figuras guardadas en reports/figs/')


if __name__ == '__main__':
    main()
