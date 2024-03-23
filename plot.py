import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = {
    'time': [0, 1, 2, 3, 4],
    'AUC_RSF': [0.77457805, 0.80593139, 0.81364628, 0.79631005, 0.65836781],
    'AUC_Discrete': [0.7498240618543209, 0.6923026052838748, 0.6315181425868449, 0.6101605621076793, 0.5855144032921811],
    'AUC_Bayesian': [0.513148, 0.475947, 0.416001, np.nan, np.nan]
}

df = pd.DataFrame(data)

plt.figure(figsize=(6, 6))  # Decrease width to make graph narrower

# Increase font sizes
plt.rcParams['font.size'] = 13  # Adjusts the default font size
plt.rcParams['axes.labelsize'] = 14  # Adjusts x and y labels font size
plt.rcParams['xtick.labelsize'] = 12  # Adjusts x-axis tick label font size
plt.rcParams['ytick.labelsize'] = 12  # Adjusts y-axis tick label font size
plt.rcParams['legend.fontsize'] = 12  # Adjusts legend font size
plt.rcParams['axes.titlesize'] = 16  # Adjusts title font size if you add a title
plt.plot(df['time'], df['AUC_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['AUC_Discrete'], marker='o', linestyle='-', color='blue', label='Discete survival model')
plt.plot(df['time'], df['AUC_Bayesian'], marker='o', linestyle='-', color='green', label='Bayesian causal model')
#plt.title('AUC over Time')
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('Time (Years)')
plt.ylabel('AUC')
plt.legend(loc='lower right')

plt.grid(True)  # Add a grid for better readability
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()



data = {
    'time': [0, 1, 2, 3, 4, 5],
    'Brier_score_RSF': [0.15334573496951814, 0.189270, 0.193621, 0.204780, 0.208777, 0.224739],
    'Brier_score_Discrete': [0.110983, 0.141267, 0.148436, 0.176767, 0.209872, 0.222184],
    'Brier_score_Bayes': [0.149836, 0.159919, 0.128879, np.nan, np.nan, np.nan]

}
# Creating a DataFrame
df = pd.DataFrame(data)



plt.figure(figsize=(6, 6))  # Decrease width to make graph narrower


plt.rcParams['font.size'] = 13  # Adjusts the default font size
plt.rcParams['axes.labelsize'] = 14  # Adjusts x and y labels font size
plt.rcParams['xtick.labelsize'] = 12  # Adjusts x-axis tick label font size
plt.rcParams['ytick.labelsize'] = 12  # Adjusts y-axis tick label font size
plt.rcParams['legend.fontsize'] = 12  # Adjusts legend font size
plt.rcParams['axes.titlesize'] = 16  # Adjusts title font size if you add a title


plt.plot(df['time'], df['Brier_score_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['Brier_score_Discrete'], marker='o', linestyle='-', color='blue', label='Discete survival model')
plt.plot(df['time'], df['Brier_score_Bayes'], marker='o', linestyle='-', color='green', label='Bayesian causal model')
#plt.title('Brier Score over Time')
plt.xlabel('Time (Years)')
plt.ylabel('Brier Score')
plt.legend(loc = 'best')

plt.grid(True)  # Add a grid for better readability
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()


aa = 0.857971*0.830846
ab = 0.857971*0.830846*0.982453
data = {
    'time': [0, 1, 2, 3, 4, 5],
    'nodiscount_Discrete': [0.9071953846, 0.8060586875, 0.7352967617, 0.6632053886,0.6148284882,0.5789819005],
    'nodiscount_RSF': [0.886757, 0.779244, 0.703352, 0.649414, 0.61716, 0.616383],
    'nodiscount_Bayes': [0.857971, aa, ab, np.nan, np.nan, np.nan]

    }
# Creating a DataFrame
df = pd.DataFrame(data)

# Adjust figure size here (width, height) to make it narrower
plt.figure(figsize=(6, 6))  # Decrease width to make graph narrower

# Increase font sizes
plt.rcParams['font.size'] = 12  # Adjusts the default font size
plt.rcParams['axes.labelsize'] = 14  # Adjusts x and y labels font size
plt.rcParams['xtick.labelsize'] = 12  # Adjusts x-axis tick label font size
plt.rcParams['ytick.labelsize'] = 12  # Adjusts y-axis tick label font size
plt.rcParams['legend.fontsize'] = 12  # Adjusts legend font size
plt.rcParams['axes.titlesize'] = 16  # Adjusts title font size if you add a title
plt.figure(figsize=(6, 6))
plt.plot(df['time'], df['nodiscount_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['nodiscount_Discrete'], marker='o', linestyle='-', color='blue', label='Discete survival model')
plt.plot(df['time'], df['nodiscount_Bayes'], marker='o', linestyle='-', color='green', label='Bayesian logistic')
#plt.title('Survival function Non Discount customers')
plt.xlabel('Time (Years)')
plt.ylim(0.5,1)
plt.ylabel('mean survival probability')
plt.legend()
plt.grid(True)  # Add a grid for better readability
plt.tight_layout()
plt.show()


# Data preparation
bb = 0.853999 * 0.841527
bc = 0.853999 * 0.841527 * 0.852274
data = {
    'time': [0, 1, 2],
    'discount_Discrete': [0.8056584128, 0.6290134692, 0.5723574354],
    'discount_RSF': [0.799636, 0.677937, 0.648282],
    'discount_Bayes': [0.853999, bb, bc]
}
df = pd.DataFrame(data)

# Adjust figure size here (width, height) to make it narrower
plt.figure(figsize=(6, 6))  # Decrease width to make graph narrower

# Increase font sizes
plt.rcParams['font.size'] = 13  # Adjusts the default font size
plt.rcParams['axes.labelsize'] = 14  # Adjusts x and y labels font size
plt.rcParams['xtick.labelsize'] = 12  # Adjusts x-axis tick label font size
plt.rcParams['ytick.labelsize'] = 12  # Adjusts y-axis tick label font size
plt.rcParams['legend.fontsize'] = 12  # Adjusts legend font size
plt.rcParams['axes.titlesize'] = 16  # Adjusts title font size if you add a title

# Plotting
plt.plot(df['time'], df['discount_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['discount_Discrete'], marker='o', linestyle='-', color='blue', label='Discrete survival model')
plt.plot(df['time'], df['discount_Bayes'], marker='o', linestyle='-', color='green', label='Bayesian logistic')

# Labels and Legend
plt.xlabel('Time (Years)')
plt.ylabel('Mean Survival Probability')
plt.legend()
plt.xticks([0, 1, 2])
plt.ylim(0.5, 1)
plt.grid(True)  # Add a grid for better readability
plt.tight_layout()

# Show plot
plt.show()