import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Category': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Value': [40, 15, 30, 10, 5, 25]
}

df = pd.DataFrame(data)

df = df.sort_values(by='Value', ascending=False) # keep bars in descending order

df['Cumulative Percentage'] = df['Value'].cumsum() / df['Value'].sum() * 100    # calculate cumulative percentage

sns.set(style="whitegrid") # set the style of seaborn

fig, ax1 = plt.subplots(figsize=(10, 6)) # create a figure and axis

# plot
ax1.bar(df['Category'], df['Value'], color='skyblue')
ax1.set_ylabel('Value', color='blue')
ax1.set_xlabel('Category')
ax1.set_title('Pareto Chart')

# create a second y-axis
ax2 = ax1.twinx()

# Plot the cumulative percentage line
ax2.plot(df['Category'], df['Cumulative Percentage'], color='green', marker='o', linestyle='-')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.set_ylabel('Cumulative Percentage', color='green')

# Show
plt.show()
