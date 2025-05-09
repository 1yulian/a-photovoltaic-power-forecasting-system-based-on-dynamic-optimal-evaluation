
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


COLORS = ['#2F5C9E', '#E37222', '#60A917', '#D80073',
          '#8E44AD', '#16A085', '#C0392B', '#7F8C8D', '#F1C40F']
ENSEMBLE_COLOR = '#E74C3C'


def plot_comparison(plot_data, xlim=None):

    actual = plot_data['actual']
    ensemble_pred = plot_data['ensemble_pred']
    model_predictions = plot_data['model_predictions']
    top_models = plot_data['top_models']
    weights = plot_data['weights']
    beta = plot_data['beta']

    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual', linewidth=1.8,
             color=COLORS[0], linestyle='-')

    for i, model in enumerate(top_models):
        plt.plot(model_predictions[model],
                 linewidth=1.8,
                 linestyle='-',
                 color=COLORS[i + 1],
                 label=f'{model} (w={weights[model]:.2f})')


    plt.plot(ensemble_pred,
             linewidth=1.8,
             linestyle='-',
             color=ENSEMBLE_COLOR,
             label=f'Ensemble (β={beta})')


    if xlim:
        plt.xlim(xlim)
        plt.title(f'Prediction Comparison ({xlim[0]}-{xlim[1]} Steps)',
                  fontsize=14, pad=18, fontweight='semibold')
    else:
        plt.title('Full Prediction Comparison',
                  fontsize=14, pad=18, fontweight='semibold')


    plt.xlabel('Time Steps', fontsize=11, labelpad=10)
    plt.ylabel('Target Value', fontsize=11, labelpad=10)
    plt.xticks(fontsize=10, rotation=30)
    plt.yticks(fontsize=10)
    plt.grid(True, color='#EDEDED', linewidth=0.8)

    legend = plt.legend(bbox_to_anchor=(1.02, 0.9),
                        frameon=True,
                        framealpha=0.9,
                        edgecolor='#FFFFFF',
                        fontsize=10)
    legend.get_frame().set_facecolor('#F5F5F5')


    ax = plt.gca()
    ax.set_facecolor('#F9F9F9')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()