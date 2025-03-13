import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(file_path, graph_title, x_label, y_label, save_path):

    # Load the data
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if not {'Step', 'Value'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'Step' and 'Value' columns.")
    
    # Plot the data
    plt.figure(figsize=(5, 5))
    plt.plot(df['Step'], df['Value'], linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.grid(False)
    plt.savefig(save_path)



if __name__ == "__main__":
    plot_csv_data("./sac_actor_loss.csv", graph_title="Actor Loss", x_label="Timestep", y_label="Loss", save_path="./sac_actor_loss.png")
    plot_csv_data("./sac_critic_loss.csv", graph_title="Critic Loss", x_label="Timestep", y_label="Loss", save_path="./sac_critic_loss.png")
    plot_csv_data("./sac_mean_reward.csv", graph_title="Mean Reward", x_label="Timestep", y_label="Reward", save_path="./sac_mean_reward.png")