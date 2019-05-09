"""
This script is developed to train and predict the network generator.
The steps involve:
    1) preparing the dataset,
    2) shuffle, split and cross-validate with k-fold
    3) train the model
    4) evaluate the model
    5) predict with the trained model
"""

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from net_gen import *


# evaluate a single mlp model
def evaluate_model(trainX, trainy, testX, testy, input_size=4, output_size=2):
    # encode targets to one-hot format
    trainy_enc = to_categorical(trainy)
    # print(trainy_enc.shape[-1])
    testy_enc = to_categorical(testy)
    # define model
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainy_enc, epochs=50, verbose=0)
    # model = svm.SVC()
    # model.fit(trainX, trainy)
    # fit model

    # evaluate the model
    _, test_acc = model.evaluate(testX, testy_enc, verbose=0)
    return model, test_acc


# evaluate a single mlp model
def evaluate_model_svm(trainX, trainy, testX, testy, input_size=4, output_size=2):
    # encode targets to one-hot format
    # trainy_enc = to_categorical(trainy)
    # print(trainy_enc.shape[-1])
    # testy_enc = to_categorical(testy)
    # define model
    model = svm.SVC(gamma='auto')

    model.fit(trainX, trainy)

    # model.fit(trainX, trainy)
    # fit model

    # evaluate the model
    test_acc = model.score(testX, testy)
    return model, test_acc



# train and evaluate the model
def train_clf(labels, features, opt='svm'):
    y, X = labels, features
    # print(X.shape)
    num_class = to_categorical(y).shape[-1]
    # cross validation estimation of performance
    scores, members = list(), list()
    # prepare the k-fold cross-validation configuration
    n_folds = 10
    kfold = KFold(n_folds, True, 1)
    for train_ix, test_ix in kfold.split(X):
        # print(train_ix)
        # print(test_ix)
        # # select samples
        trainX, trainy = X[train_ix], y[train_ix]
        testX, testy = X[test_ix], y[test_ix]
        # evaluate model
        if opt == 'mlp':
            model, test_acc = evaluate_model(trainX, trainy, testX, testy, input_size=X.shape[-1], output_size=num_class)
        if opt == 'svm':
            model, test_acc = evaluate_model_svm(trainX, trainy, testX, testy, input_size=X.shape[-1],
                                             output_size=num_class)
        print('test accuracy >%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)

        # pass

    # summarize expected performance
    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    return members, scores



def visual_dat(labels, features, fea_test):

    da = fea_test[0]
    dv = fea_test[1]
    ca = fea_test[2]
    cv = fea_test[3]

    colors = ['red', 'blue', "green"]
    classes = ['Erdos-Renyi', 'Barabasi-Albert', "to predict"]
    markers = ['^', 'v', 'x']
    y = labels

    p = np.array([fea[0] for fea in features])
    q = np.array([fea[1] for fea in features])
    s = np.array([fea[2] for fea in features])
    t = np.array([fea[3] for fea in features])
    idx = np.arange(len(y))
    ix0 = (y == 0)
    ix0 = idx[ix0]
    ix1 = (y == 1)
    ix1 = idx[ix1]

    fig, ax = plt.subplots(3, 2)

    ax[0, 0].scatter(p[ix0], q[ix0], color=colors[0], label=classes[0], marker=markers[0])
    ax[0, 1].scatter(p[ix0], s[ix0], color=colors[0], label=classes[0], marker=markers[0])
    ax[1, 0].scatter(p[ix0], t[ix0], color=colors[0], label=classes[0], marker=markers[0])
    ax[1, 1].scatter(q[ix0], s[ix0], color=colors[0], label=classes[0], marker=markers[0])
    ax[2, 0].scatter(q[ix0], t[ix0], color=colors[0], label=classes[0], marker=markers[0])
    ax[2, 1].scatter(s[ix0], t[ix0], color=colors[0], label=classes[0], marker=markers[0])

    ax[0, 0].scatter(p[ix1], q[ix1], color=colors[1], label=classes[1], marker=markers[1])
    ax[0, 1].scatter(p[ix1], s[ix1], color=colors[1], label=classes[1], marker=markers[1])
    ax[1, 0].scatter(p[ix1], t[ix1], color=colors[1], label=classes[1], marker=markers[1])
    ax[1, 1].scatter(q[ix1], s[ix1], color=colors[1], label=classes[1], marker=markers[1])
    ax[2, 0].scatter(q[ix1], t[ix1], color=colors[1], label=classes[1], marker=markers[1])
    ax[2, 1].scatter(s[ix1], t[ix1], color=colors[1], label=classes[1], marker=markers[1])

    ax[0, 0].scatter(da, dv, color=colors[2], label=classes[2], marker=markers[2])
    ax[0, 1].scatter(da, ca, color=colors[2], label=classes[2], marker=markers[2])
    ax[1, 0].scatter(da, cv, color=colors[2], label=classes[2], marker=markers[2])
    ax[1, 1].scatter(dv, ca, color=colors[2], label=classes[2], marker=markers[2])
    ax[2, 0].scatter(dv, cv, color=colors[2], label=classes[2], marker=markers[2])
    ax[2, 1].scatter(ca, cv, color=colors[2], label=classes[2], marker=markers[2])

    # plt.show()
    ax[0, 0].set_title("deg_ave vs deg_var")
    ax[0, 1].set_title("deg_ave vs clst_ave")
    ax[1, 0].set_title("deg_ave vs clst_var")
    ax[1, 1].set_title("deg_var vs clst_ave")
    ax[2, 0].set_title("deg_var vs clst_var")
    ax[2, 1].set_title("clst_ave vs clst_var")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[2, 0].legend()
    ax[2, 1].legend()

    fig.tight_layout()

    plt.savefig("dat/fea_dist.pdf")
    plt.show()
    plt.close()


# ensemble prediction
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    print(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result


def ensemble_predictions_nod(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # print(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    summed = np.reshape(summed, 2)
    summed = summed.view()
    print(summed.shape)
    # ratio {1} : {0} is correlated to the confidence of prediction
    ratio = summed[1] / summed[0]
    # argmax across classes
    # result = np.argmax(summed, axis=1)
    print("ratio is %f" %ratio)
    result = np.argmax(summed)
    return result, ratio


def ensemble_predictions_nod_sk(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # print(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # summed = np.reshape(summed, 2)
    summed = summed.view()
    print(summed.shape)
    # ratio {1} : {0} is correlated to the confidence of prediction
    ratio = summed
    # argmax across classes
    # result = np.argmax(summed, axis=1)
    print("ratio is %f" %ratio)
    result = np.argmax(summed)
    return result, ratio


def net_est(n_sample = 200):
    labels, features = dat_gen(m=n_sample)

    # # shuffle the dataset
    # idx = np.random.permutation(len(labels))
    # y, X = labels[idx], features[idx]


    # peek ajm to test
    aj_test, _ = get_test()
    fea_test = fea_gen(aj_test)
    print(fea_test)

    # # visualize the feature distribution from different net_gens
    # visual_dat(labels, features, fea_test)

    # fea_test = fea_test.reshape([4,-1])
    if (fea_test.ndim == 1):
        fea_test = np.array([fea_test])
    # predict the generator of aj_matrix

    models, scores = train_clf(labels, features)

    generator = ['Erdos-Renyi', 'Barabasi-Albert']
    result = ensemble_predictions(models, fea_test)
    print(scores)
    print("The aj matrix to predict is generated by >>> %s !" %(generator[int(result)]))

    return



# net_est(n_sample=200)





