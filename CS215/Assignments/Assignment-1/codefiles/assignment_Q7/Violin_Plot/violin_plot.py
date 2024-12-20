import pandas as pd
import matplotlib.pyplot as plt

# Generating the violin plot using in built violin plot method from matplotlib.

dataframe = pd.read_csv("gapminder_full.csv", encoding="ISO-8859-1")

population = dataframe.population
life_exp = dataframe.life_exp
gdp_cap = dataframe.gdp_cap

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

# Plot violin plot on axes 1
ax1.violinplot(dataframe.population, showmedians=True)
ax1.set_title('Population')

# Plot violin plot on axes 2
ax2.violinplot(life_exp, showmedians=True)
ax2.set_title('Life Expectancy')

# Plot violin plot on axes 3
ax3.violinplot(gdp_cap, showmedians=True)
ax3.set_title('GDP Per Cap')

plt.show()
