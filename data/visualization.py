import matplotlib.pyplot as plt

def plot_sample(sample_df):
    """Visualize hand landmark samples"""
    for k in range(len(sample_df)):
        row = sample_df.iloc[k]
        
        # Extract x and y coordinates
        x = [row[f'x{i}'] for i in range(1, 22)]
        y = [row[f'y{i}'] for i in range(1, 22)]
        
        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c='blue')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
