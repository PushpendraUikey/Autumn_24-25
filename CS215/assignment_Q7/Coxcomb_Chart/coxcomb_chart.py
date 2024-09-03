import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "year": [2004, 2022, 2004, 2022, 2004, 2022],
    "countries" : [ "Denmark", "Denmark", "Norway", "Norway","Sweden", "Sweden",],
    "sites": [4,10,5,8,13,15]
}

df= pd.DataFrame(data)


sort_order_dict = {"Denmark":1, "Sweden":2, "Norway":3, 2022:5, 2004:4,}
df = df.sort_values(by=['year','countries',], key=lambda x: x.map(sort_order_dict))

countries = df.countries.unique()
years = df.year.unique()
x = len(df.countries.unique())
sites = df.sites

# colors for each bar segment
colors = ["#973A36","#4562C5","#141936","#CC5A43","#5475D6","#2C324F",]

# Update layout for better visualization
fig, ax = plt.subplots(figsize=(5,5),facecolor = "#FFFFFF",subplot_kw=dict(polar=True) )    # make circular 
fig.tight_layout(pad=3.0)   # padding to avoid overlap

bottom = np.zeros(x)
for year in years:
    y = df[df["year"] == year]["sites"].values
    x_max = 2*np.pi
    width = x_max/len(countries)
    x_coords = np.linspace(0, x_max, len(countries), endpoint=False)
    ax.bar(x_coords, y,width= width,bottom = bottom,)
    bottom +=y

for bar, color, site in zip(ax.patches, colors, sites):     # inbuilt zip function to iterate over multiple iterables simultaneously.
    bar.set_facecolor(color)
    ax.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height()/2+ bar.get_y(),     # to put text at center of bar
        site,
         ha='center', va="center", size=8,      # horizontal and verticle alignment of text
        color = "w", weight= "light",)

ax.set_axis_off()  # Removing axis lines
ax.set_theta_zero_location("N") #setting the initial point(zero angle) to North

plt.show()