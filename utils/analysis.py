import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use('seaborn')
rcParams['figure.figsize'] = (8, 5)

def data_info(data):
    # get dataset info
    print("-"*45)
    print("Data information")
    print("-"*45)
    print(data.info())
    
def data_shape(data):
    # get dataset shape
    print("-"*11)
    print("Data shape")
    print("-"*11)
    print(data.shape)
    
def data_describe(data):
    # get dataset description
    print("-"*16)
    print("Data description")
    print("-"*16)
    print(data.describe().T)
    
def missing_values(data):
    # check for missing values in dataset
    print("-"*18)
    print("Missing values")
    print("-"*18)
    print(data.isnull().sum())
    
def word_dist(data):
    # get the word length distribution
    print("-"*25)
    print("Word length distribution")
    print("-"*25)

    word_length = data['comment_text'].str.split().apply(len)
    sns.histplot(word_length,bins=50,)
    plt.title("Word length distribution", fontsize=18)
    plt.xlabel("Number of words", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.xlim(0, 800)
    plt.savefig('images/word_dist.png')
    plt.show()
    
def toxic_types(data):
    # get the list of toxicity types
    print("-"*15)
    print("Toxicity types")
    print("-"*15)
    
    t_types = list(data.columns.values)
    t_types = t_types[2:]
    print(t_types)
    return t_types

def comments_dist(data):
    # get the distribution of comments
    print("-"*21)
    print("Comments distribution")
    print("-"*21)
    
    sum_of_row = data.iloc[:,2:].sum(axis=1)
    clean_comments = (sum_of_row==0).sum(axis=0)
    total_comments = len(data)
    toxic_comments = total_comments - clean_comments
    print(f"Total comments: {total_comments}")
    print(f"Clean comments: {clean_comments}")
    print(f"Toxic comments: {toxic_comments}")
    print()

    print(f"Percentage of clean comments: {(clean_comments/total_comments)*100:.2f}%")
    print(f"Percentage of toxic comments: {(toxic_comments/total_comments)*100:.2f}%")
    
def toxic_dist(data, t_types):
    # get the distribution of comments of each toxicity type
    print("-"*27)
    print("Toxic comments distribution")
    print("-"*27)

    y = data.iloc[:,2:].sum().values
    
    ax = sns.barplot(x=t_types, y=y)

    plt.title("Toxic comments distribution", fontsize=18)
    plt.ylabel("Number of comments", fontsize=15)
    plt.xlabel("Toxicity type", fontsize=15)

    rects = ax.patches
    labels = y
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=12)
    plt.savefig('images/toxic_dist.png')
    plt.show()
    
def toxic_class_dist(data):
    # get the distribution of each toxic class
    print("-"*25)
    print("Toxic class distribution")
    print("-"*25)

    rcParams['figure.figsize'] = (10, 5)
    fig, ax = plt.subplots(2, 3)
    data.hist(ax = ax)

    fig.suptitle('Toxic comments classification',size = 18)

    plt.savefig('images/class_dist.png')
    plt.show()
    
def toxic_class_perc(data, col):
    # get the percentage of each toxic class
    print("-"*30)
    print(f"{col} class percentage")
    print("-"*30)

    perc = data[col].value_counts(normalize=True)*100
    print(f"{round(perc, 2)}")
    print()
    
# def multi_label(data):
#     # show count of comments with multiple label
#     print("-"*22)
#     print(f"Multi-labeled comments")
#     print("-"*22)

#     rcParams['figure.figsize'] = (10, 5)
#     fig = px.histogram(data.iloc[:, 2:].sum(axis=1), barmode='group', text_auto=True)
#     fig.update_layout(bargap=0.3)
#     fig.show()

def multi_label(data):
    # show count of comments with multiple label
    print("-"*22)
    print(f"Multi-labeled comments")
    print("-"*22)

    sum_of_row = data.iloc[:,2:].sum(axis=1)
    label_count = sum_of_row.value_counts().iloc[1:]
    
    ax = sns.barplot(x=label_count.index, y=label_count.values)

    plt.title("Multi-labeled comments", fontsize=18)
    plt.ylabel("Number of comments", fontsize=15)
    plt.xlabel("Number of labels", fontsize=15)

    rects = ax.patches
    labels = label_count.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=12)
    plt.savefig('images/multi_label.png')
    plt.show()
    
def toxic_heatmap(data):
    # heatmap to show correlation between labeled class
    print("-"*22)
    print(f"Labeled class heatmap")
    print("-"*22)

    sns.heatmap(data.corr(), annot=True)
    plt.suptitle("Labeled class correlation heatmap",fontsize = 18)
    plt.xlabel("Labeled classes", fontsize=15)
    plt.ylabel("Labeled classes", fontsize=15)

    plt.savefig('images/heatmap.png')
    plt.show()

