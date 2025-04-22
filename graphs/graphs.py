import matplotlib.pyplot as plt
import os
import pandas as pd

graph_labels = {
    "episodic_return": "Episodic Returns", 
    "actor_loss": "Actor Loss", 
    "critic_loss": "Critic Loss"
}

def create_graph(approach="env_gz", graph_type="episodic_returns"):
    global graph_labels

    data_folder = f"./{approach}/{graph_type}"
    output_folder = f"./{approach}/{graph_type}/output"
    os.makedirs(output_folder, exist_ok=True)

    x_label = "Timesteps"
    y_label = graph_labels[graph_type]

    plt.figure(1)
    plt.title(graph_labels[graph_type])
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)

    for f in os.listdir(data_folder):
        if os.path.isdir(f"{data_folder}/{f}"):
            continue
        
        data = pd.read_csv(f"{data_folder}/{f}")
        data = data[['Step', 'Value']]
        algorithm = f.split('.')[0]

        title = f"{graph_labels[graph_type]} for {algorithm.upper()}"

        timesteps = data['Step']
        values = data['Value'].ewm(span=50, adjust=False).mean()
        print(approach, graph_type, algorithm, values.iloc[[-1]]
)

        plt.figure(0)
        plt.plot(timesteps, values, linestyle='-')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(f"{output_folder}/{algorithm}.png")
        plt.close()

        plt.figure(1)
        plt.plot(timesteps, values, linestyle='-', label=algorithm.upper())
    
    plt.figure(1)
    plt.legend(loc="best")
    plt.savefig(f"{output_folder}/combined.png")
    plt.close()


if __name__ == "__main__":
    approaches = ['env_gz', 'reward_shaping']
    graph_types = ['episodic_return', 'actor_loss', 'critic_loss']
    for approach in approaches:
        for graph_type in graph_types:
            create_graph(approach=approach, graph_type=graph_type)