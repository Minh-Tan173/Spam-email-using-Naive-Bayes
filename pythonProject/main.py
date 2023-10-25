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
    print(f"Ham email: {round(Ratio_of_hamEmail, 2)}%")
    print(f"Spam email: {round(Ratio_of_spamEmail, 2)}%")
        # plt.subplot(Total number of rows, Total number of columns, plot number)
    lIst = dataframe[dataframe["Label"] == 1]["length"]
    plt.subplot(2, 1, 1)
    plt.figure(figsize=(16, 8))
    plt.hist(lIst, color="blue", alpha=0.7, bins=100)
    plt.hist(dataframe[dataframe["Label"] == 0]["length"], color="red", alpha=0.7, bins = 100)
    plt.xlabel("Lines")
    plt.ylabel("Number of words in dataframe per line")
    plt.subplot(2, 1, 2)
    plt.pie(values,labels=[f"{round(Ratio_of_hamEmail, 2)}%", f"{round(Ratio_of_spamEmail, 2)}%"])  # Vẽ đồ thị hình tròn
    # plt.legend(["Valid Email", "Spam Email"], loc="upper left") # Thêm chú giải vào góc trên bên trái
    # plt.show() # In đồ thị ra màn hình
    return dataframe

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def Pretreatment(dataframe):
    cv = CountVectorizer()
    features = cv.fit_transform(dataframe["EmailText"])
    """
    Thực hiện chuyển đổi dữ liệu văn bản từ cột 'EmailText' trong dataframe thành dạng vector sử dụng CountVectorizer. 
    Kết quả của quá trình chuyển đổi được lưu trữ trong biến X_dataframeTransform.
    - Đọc tài liệu về phương pháp bad-of-words (Mục 11.2.1.1) để hiểu tại sao phải chuẩn hoá văn bản như này
      --> link: https://phamdinhkhanh.github.io/deepai-book/ch_ml/FeatureEngineering.html
    - Quy trình hoạt động của CountVertorizer.fit_transform:
      --> link: https://stackoverflow.com/questions/47898326/how-vectorizer-fit-transform-work-in-sklearn
    - features lưu trữ dòng của dataframe dưới dạng ma trận thưa, Ví dụ trực tiếp với dòng 1 (dòng đầu tiên của dataframe không tính title): 
  (0, 3558)	1 ---> từ này nằm ở cột 3558 trong từ điển được tạo bởi hàm CountVectorizer(), số 0 đại diện cho dòng bao nhiêu trong features
  (0, 8046)	1
  (0, 4359)	1
  (0, 5932)	1
  (0, 2332)	1
  (0, 1304)	1
  (0, 5549)	1
  (0, 4096)	1
  (0, 1754)	1
  (0, 3642)	1
  (0, 8509)	1
  (0, 4485)	1
  (0, 1752)	1
  (0, 2053)	1
  (0, 7661)	1
  (0, 3602)	1
  (0, 1070)	1
  (0, 8285)	1
    """
    print(features)
    X = features.toarray()
    """
    Bởi vì features là một ma trận thưa, mà thuật toán Naives Bayes lại nhận đầu vào là một ma trận dày, nên phải dùng phương thức toarray()
    để chuyển đổi X về một ma trận dày, khi này, kể cả những chứ cái không xuất hiện trong mỗi dòng cx sẽ được đánh số lần xuất hiện là 0.
    Ví dụ trực tiếp với dòng đầu tiên của X:
    [0 0 0 ... 0 0 0]: nó hiện như này vì ở cột 1,2,3 và 3 cột cuối của dòng không hề xuất hiện từ nào trong từ điển, nếu sử dụng công cụ 
    debug để xem matrix(X), ta chỉ cần kéo tới những cột 3558, 8046,4359,... tương ứng thì sẽ thấy chỉ số là 1 (vì những từ này xuất hiện trong
    từ điển, những từ khác không có nên mới có chỉ số là 0)    
    """
    # print(X)
    y = dataframe["Label"]
    # Extract and separation data
    X_train, X_test, y_train, y_test  = train_test_split(X,y, train_size= 0.8)
    print()
    word = cv.get_feature_names_out()
    print(word)  # in ra từ điển từ được tạo bởi phương thức CountVectorizer(), tất cả 8678 từ (8678 collumns)
    print(f"X_train = {X_train.shape}, datatype: {type(X_train)}") # trả về 1 tuple kích thước của ma trận X_train. Với 4457 là số hàng, 8679 là số cột
    print(f"X_test = {X_test.shape}, datatype: {type(X_test)}")
    print(f"y_train = {y_train.shape}, datatype: {type(y_train)}")
    print(f"y_test = {y_test.shape}, datatype: {type(y_test)}")
    print()
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
        """
        - alpha: smoothing parameter. Tham số này rất quan trọng trong tính toán, nếu không có nó, thuật toán rất có thể sẽ sai
        - Ví dụ: trường hợp 1 đội bóng A phải đá 9 trận đấu, 8 trận trước họ đều thua, tính xác suất để trận cuối thắng, nếu không
        có thông số alpha để làm chuẩn xác xác suất, thì với cách tính xác suất: P(win) =  win/(lost+win) = 0/(8+0) = 0, khả năng
        họ thắng sẽ là 0 --> ko đúng, xác suất thắng phải có (bất kể nhỏ hay lớn), từ đó thông số alpha (còn được gọi là laplace smoothing)
        được sinh ra, bằng việc sử dụng công thức mới P(win) = (win+alpha)/{(lost+alpha) + (win +alpha)} = (0+1)/{(8+1)+(0+1)} = 1/10 --> độ chính xác đã cao hơn
        - Nếu muốn hiểu rõ hơn thì xem cái này đi, một anh Ấn Độ râu quai nón sẽ giảng đầy đủ cho bm (.) 8' --> https://www.youtube.com/watch?v=IhJlLTHzPXQ&t=42s
        """

    def fit(self, X, y): # X = X_train, y= y_train tương ứng
        self.X = X
        self.y = y
        self.classes = np.unique(y) # Trích xuất và lưu trữ các label của y_train, cụ thể là [0 1]
        self.parameters = {}
        for _, c in enumerate(self.classes):
            print(f"c in this loop = {c}, type: {type(c)}")
            X_c = X[np.where(y == c)]
            self.parameters[f"phi_{str(c)}"] = len(X_c) / len(X)
            print(self.parameters[f"phi_{str(c)}"])
            print(len(X_c))
            """
            phi_str(c): xác suất tiên nghiệm của một lớp (trong pj này, là 2 lớp [0 1] tương ứng với 2 lớp ["spam", "ham"])
            --> nói theo ngôn ngữ loài người, là tỉ lệ sủa spam/(spam+ham) và ham/(spam+ham)
                len(X_c): số lượng của mỗi lớp xuất hiện trong dataframe 
            --> nói theo ngôn ngữ loài người, là tổng số dòng có label "spam" và tổng số dòng có label "ham" trong 80% dataframe (bài này lấy 80% dataframe để training)
                len(X): tổng số lượng tất cả các lớp xuất hiện trong dataframe
            --> nói theo ngôn ngữ loài người, là tổng số dòng của 80% dataframe, hay chính là len("sum") + len("ham")
            """
            self.parameters[f"theta_{str(c)}"] = (X_c.sum(axis=0) + self.alpha) / (np.sum(X_c.sum(axis=0)  + self.alpha))
            # print(np.sum(X_c.sum(axis=0)))
            """
            theta_str(c): Xác xuất mỗi đặc trưng xuất hiện trong lớp c
            ---> nói theo ngôn ngữ loài người là xác suất xuất hiện của từng chữ cái trong spam và ham
            X_c.sum(axis=0): tổng số lần xuất hiện của từng đặc trưng trong lớp c ("spam" và "ham")
            --> nói theo ngôn ngữ loài người: tổng số lần xuất hiện của 1 từ nào đó trong lớp tương ứng
                (np.sum(X_c.sum(axis=0)
            np.sum(X_c.sum(axis=0): tính tổng số lần xuất hiện của tất cả các tính năng trong tập dữ liệu X_c, không dùng len(X_c)
            như công thức trên vì len(X_c) không tính tới việc một tính năng có thể xuất hiện nhiều lần trong một điểm dữ liệu
            * Lưu ý: "một điểm dữ liệu" tức là 1 hàng trong dataframe (hay còn là 1 email trong file spam.csv) 
                     "một đặc trưng" tức là một từ trong dataframe
                     "một lớp" tức là một label trong dataframe (ở bài này cụ thể là 2 label "spam", "ham" lần lượt là [0, 1])
                     "self.alpha": tăng độ chính xác cho thuật toán (thử bỏ tham số alpha đi, độ chính xác từ 98% sẽ tụt về 13% ngay)
            """

    def predict(self, X):
        predictions = []
        for x in X:
            phi_list = []
            for _, c in enumerate(self.classes): # yêu cầu chương trình lặp lần lượt mỗi label (có 2 lần lặp, lần 1 tương ứng với c = 0, lần 2 c = 1, với 0,1 là spam,ham)
                phi = np.log(self.parameters["phi_" + str(c)])
                theta = np.sum(np.log(self.parameters["theta_" + str(c)]) * x)
                phi_list.append(phi + theta)
            predictions.append(self.classes[np.argmax(phi_list)])
        return predictions
    """
    x (x in X): x đại diện cho từng dòng trong dataframe (phần dc dùng để test - 20%) được chuẩn hoá dưới dạng ma trận dày --> lệnh for x in X cho chương trình lặp
    lần lượt qua mỗi dòng trong dataframe 
    phi: giá trị logarit của phi_0 và phi_1 (phi_0 và phi_1 chính là tỉ lệ của spam trong dataframe và ham trong dataframe --> tính ở phương thức fit bên trên)
    
    
    """


def trainModel(X_train, X_test, y_train, y_test):
    print()


if __name__ == "__main__":
    main()
