from matplotlib.pyplot import figure, subplot, plot, title, xlabel, ylabel, legend, tight_layout, show
from sklearn.metrics import classification_report

#-----------------------------------------------------------------------------

def perf(anc) : 

    # Plot of performances 

    figure()

    subplot(2, 1, 1)
    plot(anc.history['accuracy'], label='training accuracy', color = 'darkblue')
    plot(anc.history['val_accuracy'], label='test accuracy', color = 'magenta')
    title('Accuracy')
    xlabel('epochs')
    ylabel('accuracy')
    legend()

    subplot(2, 1, 2)
    plot(anc.history['loss'], label='training loss', color = 'darkblue')
    plot(anc.history['val_loss'], label='test loss', color = 'magenta')
    title('Loss')
    xlabel('epochs')
    ylabel('loss')
    legend()

    tight_layout()
    show()

#-----------------------------------------------------------------------------

def ratio_kaggle (y_test_tc, X_test, model) :

    # Here's a function that will give the score that we can see on kaggle based on the testing dataset 

    true = y_test_tc.argmax(axis=1)

    print("True codes : ", true)
    print("Number of true codes : ", len(true))

    predict = model.predict(X_test).argmax(axis=1)

    print("Predictions : ", predict)
    print("Number of predictions : ", len(predict))

    right = 0 

    for i in range(len(true)) : 
        if predict[i] == true[i] :  
            right += 1 

    print("Number of right : ", right)
    print("Number of elements : ", len(true))

    print("Ratio : ", right/len(true))

#-----------------------------------------------------------------------------

def network (model, y_test_tc, X_test, label_names, nbr_class) : 

    # Evaluate the network

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test) 
    print(classification_report(y_test_tc.copy().argmax(axis=1),
        predictions.argmax(axis=1), target_names=label_names, labels=range(nbr_class)))