import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2 to calculate BMI and overweight
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# 3 to normalize cholesterol and glucose level
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4
    def draw_cat_plot():
    # Create a DataFrame for the categorical plot
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])


    # 6 group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    

    # 7 # Rename column for proper plotting
    df_cat.rename(columns={"variable": "Feature", "value": "Level"}, inplace=True)
    
    # Draw the categorical plot
    fig = sns.catplot(x="Feature", hue="Level", col="cardio", data=df_cat, kind="count").fig
    
    # 8
    plt.show()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
]
    # 12 Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate the mask for the upper triangle
    mask = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', cbar=True, linewidths=0.5)
    plt.show()



    # 16
    fig.savefig('heatmap.png')
    return fig


 # Calling the functions to draw the plots
draw_cat_plot()
draw_heat_map()