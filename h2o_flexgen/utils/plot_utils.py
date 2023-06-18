import matplotlib
import matplotlib.pyplot as plt
import os


linestyles = ["solid", "dashed", "dashdot", "dotted", (0, (3,5,1,5,1,5))]
petals_linestyles = ["solid", "solid", "dotted", "dotted", 
                     "dotted", "dotted", "dotted", "dotted",
                     "dotted", "dotted", "dotted", "dotted",
                     "dotted", "dotted", "dotted", "dotted",
                     "dotted", "dotted", "dotted", "dotted",
                     "dotted", "dotted", "dotted", "dotted",
                    ]

method2color_dict = {"FlexGen (c)": "C0",
                     "FlexGen": "C1",
                     "DeepSpeed": "C2",
                     "HuggingFace": "C3",
                     "Petals": "C4"}
method2style_dict = {"FlexGen (c)": "dashed",
                     "FlexGen": "solid",
                     "DeepSpeed": "solid",
                     "HuggingFace": "solid",
                     "Petals": "solid"}
ct = 0
def method2color(name):
    global ct
    if name not in method2color_dict:
        method2color_dict[name] = f"C{ct}"
        ct += 1
    return method2color_dict[name]


def method2style(name):
    prefix = name.split(" ")[0]
    if name == "FlexGen (c)": prefix = name
    if prefix not in method2style_dict:
        method2style_dict[prefix] = "dotted"
    return method2style_dict[prefix]


def method2order(name):
    if name.startswith("FlexGen (c)"):
        return 0
    elif name.startswith("FlexGen"):
        return 1
    elif name.startswith("DeepSpeed"):
        return 2
    elif name.startswith("HuggingFace"):
        return 3
    elif name.startswith("Petals"):
        return 4
    else:
        return 5


def plot_common(data, ax, xlabel, xmin, ytick, title, legend=True):
    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    y_max = 0
    for i, method in enumerate(methods):
        curve = data[method]
        xs_, ys_ = zip(*curve.items())
        xy = sorted(zip(xs_, ys_))
        xs_ = [x for x, _ in xy]
        ys_ = [y for _, y in xy]
        xs = []
        ys = []
        for x, y in zip(xs_, ys_):
            if len(xs) == 0 or y >= ys[-1]:
                xs.append(x)
                ys.append(y)

        #print(method)
        #print(xs)
        #print(ys)

        curve = ax.plot(xs, ys, color=method2color(method), marker='*', linestyle=method2style(method), linewidth=4, markersize=20)
        curves.append(curve[0])
        legends.append(method.replace("HuggingFace", "Accelerate"))

        y_max = max(y_max, *ys)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(xmin=xmin)
    ax.set_yticks(ytick)
    ax.set_title(title, fontsize=25)

    #if "175B" in title:
    #    ax.text(2 ** 10, 2**-7, title, fontsize=25)
    #elif "30B" in title:
    #    ax.text(2 ** 9, 2**-2, title, fontsize=25)

    ax.grid()

    if legend:
        ax.legend(curves, legends, fontsize=25, loc="upper center",
                  bbox_to_anchor=(1.09, 1.25), ncol = 4)


def plot(data1, data2, output_dir, name, pdf=True):
    fig, axs = plt.subplots(1, 2)

    datas = [data1, data2]
    xlabels = ["Latency (s)", "Latency (s)"]
    lb1 = min([min(dic) for dic in data1.values()]) / 1.1
    lb2 = min([min(dic) for dic in data2.values()]) / 1.1
    xmins = [lb1, lb2]
    yticks = [
        [2**0, 2**-2, 2**-4, 2**-6, 2**-8],
        [2**3, 2**1,  2**-1, 2**-3,],
    ]
    titles = ["OPT-175B", "OPT-30B"]
    legend = True
    for data, ax, xlabel, ytick, title, xmin in zip(datas, axs, xlabels, yticks, titles, xmins):
        plot_common(data, ax, xlabel, xmin, ytick=ytick, title=title, legend=legend)
        legend = False

    fig.text(0.07, 0.5, "Generation throughput (token/s)", va='center', rotation='vertical', fontsize=25)
    
    if pdf:
        output = os.path.join(output_dir, f"{name}.pdf")
    else:
        output = os.path.join(output_dir, f"{name}.png")
    
    figure_size = (18, 7)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


def petals_plot_common(data, ax, xlabel, ylabel, xmin, legend=True, offset=1):
    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    y_max = 0
    for i, method in enumerate(methods):
        curve = data[method]
        xs_, ys_ = zip(*curve.items())
        xy = sorted(zip(xs_, ys_))
        xs_ = [x for x, _ in xy]
        ys_ = [y for _, y in xy]
        xs = []
        ys = []
        for x, y in zip(xs_, ys_):
            if len(xs) == 0 or y >= ys[-1]:
                xs.append(x)
                ys.append(y)
        curve = ax.plot(xs, ys, color=method2color(method), marker='*',
                        linestyle=method2style(method), linewidth=3, markersize=13)
        curves.append(curve[0])
        legends.append(method)

        y_max = max(y_max, *ys)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    # ax.set_xscale("log", base=2)
    # ax.set_yscale("log", base=2)
    ax.set_xlim(xmin=xmin)
    ax.grid()

    # ax.legend(curves, legends, fontsize=25, loc="upper left")
    if legend:
        ax.legend(curves, legends, fontsize=25, loc="upper center",
                  bbox_to_anchor=(1.05, offset), ncol = 2)


def petals_plot(data1, data2, output_dir, name, pdf=True, offset=1):
    fig, axs = plt.subplots(1, 2)

    datas = [data1, data2]
    xlabels = ["Output sequence length", "Output sequence length"]
    ylabels = ["Full generation latency (s)", "Throughput per GPU (token/s)"]
    xmins = [0, 0]
    legend = True
    for data, ax, xlabel, ylabel, xmin in zip(datas, axs, xlabels, ylabels, xmins):
        petals_plot_common(data, ax, xlabel, ylabel, xmin, legend, offset=offset)
        legend = False

    fig.text(0.07, 0.5, "", va='center', rotation='vertical', fontsize=25)
    
    if pdf:
        output = os.path.join(output_dir, f"{name}.pdf")
    else:
        output = os.path.join(output_dir, f"{name}.png")
    
    figure_size = (18, 7)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


