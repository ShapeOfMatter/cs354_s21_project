import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def create_plots(stats):
    models = list(stats.keys())
    info = list(stats[models[0]].keys())
    for stat in info:
        bar_plot_list = []
        for model in models:
            to_plot = stats[model][stat]
            if type(to_plot) == list:
                plt.plot(to_plot,label = model)
                plt.xlabel('epoch')
                plt.ylabel(stat)
                plt.legend(loc = 'best')
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            else:
                bar_plot_list.append(to_plot)
                pass #TODO: add bar graph
        if len(bar_plot_list) > 0:
            plt.bar(models,bar_plot_list)
            plt.ylabel(stat)
        plt.show()