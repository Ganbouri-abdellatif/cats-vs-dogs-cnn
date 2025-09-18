import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def plot_history(history):
    import pandas as pd
    results = pd.DataFrame(history.history)

    fig = px.line(results, y=[results['accuracy'], results['val_accuracy']],
                  template="seaborn", color_discrete_sequence=['#fad25a','red'])
    fig.update_layout(title='Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.show()

    fig = px.line(results, y=[results['loss'], results['val_loss']],
                  template="seaborn", color_discrete_sequence=['#fad25a','red'])
    fig.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()

def plot_class_distribution(class_names, counts):
    import plotly.express as px
    px.pie(names=class_names, values=counts).show()
