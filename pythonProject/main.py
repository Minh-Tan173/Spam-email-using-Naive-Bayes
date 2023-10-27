import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def main():
    # Data Prepare
    df = dataPrepairing()

    #Pretreatment
    X_train, X_test, y_train, y_test = Pretreatment(df)

    # Train model
    NB_classifier = CustomMultinomialNB()
    NB_classifier.fit(X_train, y_train)
    y_predict_test = NB_classifier.predict(X_test)

    # accuracy report
    print()
    print(classification_report(y_test, y_predict_test))


def dataPrepairing():
    # Create dataframe
    dataframe = pd.read_csv(r"C:\Users\admin\OneDrive\Máy tính\pythonProject\spam.csv", encoding="utf-8")
    dataframe["Label"] = (dataframe["Label"] == "ham").astype(int)
    dataframe["length"] = dataframe["EmailText"].apply(len)
    List_column = list(dataframe.columns)
    List_column[0], List_column[1] = List_column[1], List_column[0]
    dataframe = dataframe.reindex(columns=List_column)
    print(dataframe.head(10))

    # Draw a graph and show it
    values = dataframe["Label"].value_counts()
    Ratio_of_spamEmail = (values[0] / (values[0] + values[1])) * 100
    Ratio_of_hamEmail = (values[1] / (values[0] + values[1])) * 100
    print(f"Ham email: {round(Ratio_of_hamEmail, 2)}%")
    print(f"Spam email: {round(Ratio_of_spamEmail, 2)}%")
    plt.figure(figsize=(16, 7))
    plt.subplot(2, 2, 1)
    plt.suptitle("Statistics of the two labels Spam and Ham")
    plt.hist(dataframe[dataframe["Label"] == 1]["length"], color="blue", alpha=0.7, bins=100)
    plt.title("Number of letters on each line of Ham label")
    plt.xlabel("Lines")
    plt.ylabel("Words/line")
    plt.subplot(2, 2, 2)
    plt.hist(dataframe[dataframe["Label"] == 0]["length"], color="red", alpha=0.7, bins=100)
    plt.title("Number of letters on each line of Spam label")
    plt.xlabel("Lines")
    plt.ylabel("Words/line")
    plt.subplot(2, 2, 3)
    plt.pie(values,autopct="%0.2f%%" , labels=["Ham", "Spam"])  # Vẽ đồ thị hình tròn
    plt.show()

    return dataframe


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def Pretreatment(dataframe):
    cv = CountVectorizer()
    features = cv.fit_transform(dataframe["EmailText"])
    print(features)
    X = features.toarray()
    y = dataframe["Label"]
    # Extract and separation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print()
    print(f"X_train = {X_train.shape}, datatype: {type(X_train)}")
    print(f"X_test = {X_test.shape}, datatype: {type(X_test)}")
    print(f"y_train = {y_train.shape}, datatype: {type(y_train)}")
    print(f"y_test = {y_test.shape}, datatype: {type(y_test)}")
    print()
    word = cv.get_feature_names_out()
    print(word)
    return X_train, X_test, y_train, y_test


class CustomMultinomialNB:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):  # X = X_train, y= y_train tương ứng
        self.X = X
        self.y = y
        self.classes = np.unique(y)  # Trích xuất và lưu trữ các label của y_train, cụ thể là [0 1]
        self.parameters = {}
        for _, c in enumerate(self.classes):
            print(f"c in this loop = {c}, type: {type(c)}")
            X_c = X[np.where(y == c)]
            self.parameters[f"phi_{str(c)}"] = len(X_c) / len(X)
            print(self.parameters)
            self.parameters[f"theta_{str(c)}"] = (X_c.sum(axis=0) + self.alpha) / (np.sum(X_c.sum(axis=0) + self.alpha))


    def predict(self, X): # X = X_test tương ứng
        predictions = []
        for x in X:
            phi_list = []
            for _, c in enumerate(self.classes):
                phi = np.log(self.parameters[f"phi_{str(c)}"])
                theta = np.sum(np.log(self.parameters[f"theta_{str(c)}"]) * x)
                phi_list.append(phi + theta)
            predictions.append(self.classes[np.argmax(phi_list)])
        return predictions


if __name__ == "__main__":
    main()
