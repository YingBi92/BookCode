import numpy as np

def evalAccuracy(toolbox, individual, x_train, y_train):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    correctNum = 0.0
    for i in range(0, len(y_train)):
        grey = np.array(x_train[i, :, :])
        output = func(grey)
        if output <= 0 and y_train[i] == 0:
            correctNum += 1.0
        elif output > 0 and y_train[i] == 1:
            correctNum += 1.0
    accuracy = round(100*correctNum / len(y_train), 2)
    return accuracy,
