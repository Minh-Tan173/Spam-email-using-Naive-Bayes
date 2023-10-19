import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def main():
    df = dataPrepairing()
    X_train, X_test, y_train, y_test  = Pretreatment(df)

    # Train model
    NB_classifier = CustomMultinomialNB()
    # print(np.unique(y_train))
    NB_classifier.fit(X_train, y_train)
    y_predict_test = NB_classifier.predict(X_test)

    # accuracy
    print()
    print(classification_report(y_test, y_predict_test))

def dataPrepairing():
    # Create dataframe
    dataframe = pd.read_csv(r"C:\Users\admin\OneDrive\Máy tính\pythonProject\spam.csv", encoding="utf-8")
    dataframe["Label"] = (dataframe["Label"] == "ham").astype(int)
    """
    Chọn cột "class" trong dataframe, so sánh từng giá trị trong cột với "Label",
    nếu giá trị nào == "ham" thì trả về True, ngược lại là False. Sử dụng phương thức .astype(int) 
    để chuyển đổi các giá trị logic kiểu bool về số nguyên, True ứng với 1 và ngược lại
    """
    dataframe["length"] = dataframe["EmailText"].apply(len)
    # print(dataframe.head(10))
    List_column = list(dataframe.columns) # Tạo 1 list tên là List_column chứa giá trị tên của lần lượt từng column
    List_column[0], List_column[1] = List_column[1], List_column[0] # Thực hiện vc đổi index 2 columns EmailText và Label trong List_column
    dataframe = dataframe.reindex(columns= List_column) # Thực hiện vc đổi vị trí 2 columns EmailText và Label trong dataframe
    print(dataframe.head(10)) # Show dataframe ra màn hình, head(10) có nghĩa là in ra 10 dòng đầu của dataframe, nếu head() thì mặc định là 5

    # Draw a graph and show it
    values = dataframe["Label"].value_counts() # Ở cột Label trong dataframe, in ra chuỗi chứa số lượng các giá trị duy nhất
    Ratio_of_spamEmail = (values[0]/(values[0] + values[1])) * 100
    Ratio_of_hamEmail = (values[1]/(values[0] + values[1])) * 100
    print(f"Ham email: {Ratio_of_hamEmail}%")
    print(f"Spam email: {Ratio_of_spamEmail}%")
        # plt.subplot(Total number of rows, Total number of columns, plot number)
    ListOfHam = dataframe[dataframe["Label"] == 1]["length"].tolist()
    ListOfSpam = dataframe[dataframe["Label"] == 0]["length"].tolist()
    minLenHam, minLenSpam = min(np.arange(0, len(ListOfHam)+1000)), min(np.arange(0, len(ListOfSpam)+1000))
    maxLenHam, maxLenSpam = max(np.arange(0, len(ListOfHam)+1000)), max(np.arange(0, len(ListOfSpam)+1000))
    plt.subplot(2, 2, 1)
    plt.hist(dataframe[dataframe["Label"] == 1]["length"], color="blue", alpha=0.7, bins=100)
    plt.subplot(2, 2, 2)
    plt.hist(dataframe[dataframe["Label"] == 0]["length"], color="red", alpha=0.7, bins = 100)
    plt.subplot(2, 2, 3)
    plt.pie(values,labels=[f"{np.floor(Ratio_of_hamEmail)}%", f"{np.ceil(Ratio_of_spamEmail)}%"])  # Vẽ đồ thị hình tròn
    # plt.legend(["Valid Email", "Spam Email"], loc="upper left") # Thêm chú giải vào góc trên bên trái
    # plt.show() # In đồ thị ra màn hình

    return dataframe

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def Pretreatment(dataframe):
    cv = CountVectorizer()
    X_dataframeTransform = cv.fit_transform(dataframe["EmailText"])
    """
    Thực hiện chuyển đổi dữ liệu văn bản từ cột 'EmailText' trong dataframe thành dạng vector sử dụng CountVectorizer. 
    Kết quả của quá trình chuyển đổi được lưu trữ trong biến X_dataframeTransform.
    - Đọc tài liệu về phương pháp bad-of-words (Mục 11.2.1.1) để hiểu tại sao phải chuẩn hoá văn bản như này
      --> link: https://phamdinhkhanh.github.io/deepai-book/ch_ml/FeatureEngineering.html
    - Quy trình hoạt động của CountVertorizer.fit_transform:
      --> link: https://stackoverflow.com/questions/47898326/how-vectorizer-fit-transform-work-in-sklearn
    """
    X = X_dataframeTransform.toarray()

    y = dataframe["Label"]
    # Extract and separation data
    X_train, X_test, y_train, y_test  = train_test_split(X,y, train_size= 0.8)
    print(f"X_train = {X_train.shape}") # trả về 1 tuple kích thước của ma trận X_train. Với 4457 là số hàng, 8679 là số cột
    print(f"X_test = {X_test.shape}")
    print(f"y_train = {y_train.shape}")
    print(f"y_test = {y_test.shape}")
    """
    
    """
    return X_train, X_test, y_train, y_test

class CustomMultinomialNB:
    """
    Xem video này để hiểu về thuật toán MultinomialNB:
    --> https://www.youtube.com/watch?v=rdJ_CMYLiqY (xem cái này trước)
    --> https://www.youtube.com/watch?v=p-nuCQ_VmN4 (xem cái này sau)

    """
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y): # X = X_train, y= y_train tương ứng
        self.X = X
        self.y = y
        self.classes = np.unique(y) # Trích xuất và lưu trữ các label của y_train, cụ thể là [0 1]
        self.parameters = {}
        for _, c in enumerate(self.classes):
            print(f"c in this loop = {c}, type: {type(c)}")
            X_c = X[np.where(y == c)]
            self.parameters[f"phi_{str(c)}"] = len(X_c) / len(X)
            """
            phi_{str(c)}: xác suất tiên nghiệm của một lớp (trong pj này, là 2 lớp [0 1] tương ứng với 2 lớp ["spam", "hám"])
            len(X_c): số lượng của mỗi lớp xuất hiện trong dataframe 
            len(X): tổng số lượng tất cả các lớp xuất hiện trong dataframe
            """
            self.parameters["theta_" + str(c)] = (X_c.sum(axis=0) + self.alpha) / (np.sum(X_c.sum(axis=0) + self.alpha))

    def predict(self, X):
        predictions = []
        for x in X:
            phi_list = []
            for _, c in enumerate(self.classes):
                phi = np.log(self.parameters["phi_" + str(c)])
                theta = np.sum(np.log(self.parameters["theta_" + str(c)]) * x)
                phi_list.append(phi + theta)
            predictions.append(self.classes[np.argmax(phi_list)])
        return predictions


from sklearn.metrics import classification_report
def trainModel(X_train, X_test, y_train, y_test):
    print()


if __name__ == "__main__":
    main()
