import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import streamlit as st
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#  sets the configuration options
st.set_page_config(page_title="ML Model",
                   layout="wide",
                   page_icon="person")
#  creates a sidebar
with st.sidebar:
    selected = option_menu('Options',
                           ['Choose csv file'],
                           icons=['cloud-upload'],
                           default_index=0)

# ----------------------------------------------------------------------------------------------------------

if selected == 'Choose csv file':

    st.title('Choose csv file')

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    if uploaded_files:

        def remove_col(df, col):
            df = df.drop(columns=col)
            return df


        def remove_outliers(df, col):
            upper_limit = df[col].mean() + 2 * df[outlier_col].std()
            lower_limit = df[col].mean() - 2 * df[outlier_col].std()
            print(upper_limit)
            print(lower_limit)

            # find the outliers
            df = df.loc[(df[col] < upper_limit) & (df[col] > lower_limit)]
            return df

        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            invoice_df = df.head(5)
            ui.table(data=invoice_df, maxHeight=300)
            # col1, col2 = st.columns(2)

        def plot_confusion_matrix(true_labels, predicted_labels):
            # Tính toán confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)

            # Vẽ confusion matrix bằng seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.title("Confusion Matrix")
            st.pyplot(plt)


        def predict_single_variable(user_input1):
            X = df[user_input1[0]].values.reshape(-1, 1)  # Biến độc lập
            y = df[user_input1[1]].values  # Biến phụ thuộc

            model = LinearRegression()
            model.fit(X, y)

            diab_diagnosis1 = model.predict(X)

            return diab_diagnosis1


        def predict_knn_single_variable(user_input1):
            X = df[user_input1[0]].values.reshape(-1, 1)  # Biến độc lập
            y = df[user_input1[1]].values  # Biến phụ thuộc

            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X, y)

            diab_diagnosis = model.predict(X)

            return diab_diagnosis


        def predict_logistic_single_variable(user_input1):
            X = df[user_input1[0]].values.reshape(-1, 1)  # Biến độc lập
            y = df[user_input1[1]].values  # Biến phụ thuộc

            model = LogisticRegression()
            model.fit(X, y)

            diab_diagnosis = model.predict(X)

            return diab_diagnosis


        def predict_decision_tree_single_variable(user_input1):
            X = df[user_input1[0]].values.reshape(-1, 1)  # Biến độc lập
            y = df[user_input1[1]].values  # Biến phụ thuộc

            model = DecisionTreeRegressor()
            model.fit(X, y)

            diab_diagnosis = model.predict(X)

            return diab_diagnosis


        tab1, tab2, tab3 = st.tabs(["Resolve Null Values", "Remove Columns", "Outliers"])

        with tab1:
            st.header("Resolve Null Values")
            st.write("Choose column to see null values")
            null_col = st.selectbox("Independent variable", (df.columns), index=None, placeholder="Choose value",
                                    key="null-value")
            if st.button('See null values'):
                null_values = df[null_col].isnull().sum()
                st.write(null_values)
                # if null_values != 0:
                #     # sdddddddddddddddddddddddddddd

        with tab2:
            st.write("Remove column")
            # Đặt tên khác cho biến remove_col
            # unwanted_col = st.selectbox("Remove unwanted column", df.columns, index=None, placeholder="Choose value")
            unwanted_col = st.multiselect("Remove unwanted column", df.columns, placeholder="Choose value",
                                          default=None)
            st.write("Selected column:", unwanted_col)
            if st.button('Remove'):
                df = remove_col(df, unwanted_col)
                # Chọn lại 5 hàng đầu tiên từ DataFrame đã được cập nhật
                invoice_df_updated = df.head(5)

                # Hiển thị lại bảng với 5 hàng đầu tiên của DataFrame đã cập nhật
                ui.table(data=invoice_df_updated, maxHeight=300)

        with tab3:
            st.header("Remove Outliers")
            st.write("Choose column to see outliers")
            outlier_col = st.selectbox("Independent variable", (df.columns), index=None, placeholder="Choose value",
                                       key="outlier")
            if st.button('See outliers'):
                arr = df[outlier_col].values
                fig, ax = plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)
            if st.button('Remove Outlier using Z-score method'):
                df = remove_outliers(df, outlier_col)
                st.write(df.describe())
                arr = df[outlier_col].values
                fig, ax = plt.subplots()
                ax.hist(arr, bins=20)
                st.pyplot(fig)

        # with tab3:
        #     with tab3:
        #         st.write("Drop Missing Values")
        #         if st.button('Drop Missing Values'):
        #             df.dropna(inplace=True)
        #             st.write("Missing Values have been dropped.")
        #             st.write("Updated DataFrame:")
        #             st.write(df)

        with st.container(border=True):
            st.write("Single Variable")

            Independent1 = st.selectbox("Independent variable", (df.columns), index=None, placeholder="Choose value")

            Dependent1 = st.selectbox("Dependent variable", (df.columns), index=None, placeholder="Choose value")

            MH = st.selectbox("Model", ("Linear Regression", "KNN", "Logistics regression", "Decision Tree"),
                              index=None, placeholder="Choose model")

            if MH == "Linear Regression":
                if st.button('RESULT'):
                    user_input1 = [Independent1, Dependent1]
                    diab_diagnosis1 = predict_single_variable(user_input1)

                    plot_confusion_matrix(user_input1[1], diab_diagnosis1)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[user_input1[0]], df[user_input1[1]], color='blue', label='Actual data')
                    plt.plot(df[user_input1[0]], diab_diagnosis1, color='red', label='Predicted data')
                    plt.xlabel(user_input1[0])
                    plt.ylabel(user_input1[1])
                    plt.title('Linear Regression')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

            if MH == "KNN":
                if st.button('RESULT'):
                    user_input1 = [Independent1, Dependent1]
                    diab_diagnosis1 = predict_knn_single_variable(user_input1)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[user_input1[0]], df[user_input1[1]], color='blue', label='Actual data')
                    plt.plot(df[user_input1[0]], diab_diagnosis1, color='red', label='Predicted data')
                    plt.xlabel(user_input1[0])
                    plt.ylabel(user_input1[1])
                    plt.title('KNN Regression')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

            if MH == "Logistics regression":
                if st.button('RESULT'):
                    user_input1 = [Independent1, Dependent1]
                    diab_diagnosis1 = predict_logistic_single_variable(user_input1)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[user_input1[0]], df[user_input1[1]], color='blue', label='Actual data')
                    plt.plot(df[user_input1[0]], diab_diagnosis1, color='red', label='Logistic Regression')
                    plt.xlabel(user_input1[0])
                    plt.ylabel(user_input1[1])
                    plt.title('Logistic Regression')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)
            if MH == "Decision Tree":
                if st.button('RESULT'):
                    user_input1 = [Independent1, Dependent1]
                    diab_diagnosis1 = predict_decision_tree_single_variable(user_input1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(df[user_input1[0]], df[user_input1[1]], color='blue', label='Actual data')
                    plt.plot(df[user_input1[0]], diab_diagnosis1, color='red', label='Predicted data')
                    plt.xlabel(user_input1[0])
                    plt.ylabel(user_input1[1])
                    plt.title('Decision Tree Regression')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)
        st.write("Data Types:")
        st.write(df.dtypes)

        show_info = st.button("Show Information")
        if show_info:
            st.write("Missing Values:")
            st.dataframe(df.isna().sum())

            st.write("Duplicated Rows:")
            st.write(df[df.duplicated()])

            st.write("Data Distribution:")
            st.write(df.describe())

        # Add the variable selection and plot code here
        selected_variable = st.selectbox("Select a variable", df.columns)
        selected_chart_type = st.selectbox("Select a chart type",
                                           ["Histogram", "Countplot", "Line plot", "Boxplot", "Violin plot",
                                            "Pie chart"])

        if selected_variable:
            if df[selected_variable].dtype in ['int64', 'float64']:  # Kiểm tra xem kiểu dữ liệu có phải là số không
                if selected_chart_type == "Histogram":
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x=selected_variable, bins=20, kde=True)
                    plt.xlabel(selected_variable)
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    st.pyplot(plt)
                elif selected_chart_type == "Line plot":
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=df, x=df.index, y=selected_variable)
                    plt.xlabel('Index')
                    plt.ylabel(selected_variable)
                    plt.title('Line Plot')
                    plt.grid(True)
                    st.pyplot(plt)
                elif selected_chart_type == "Boxplot":
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, y=selected_variable)
                    plt.ylabel(selected_variable)
                    plt.title('Boxplot')
                    plt.grid(True)
                    st.pyplot(plt)
                elif selected_chart_type == "Violin plot":
                    plt.figure(figsize=(10, 6))
                    sns.violinplot(data=df, y=selected_variable)
                    plt.ylabel(selected_variable)
                    plt.title('Violin plot')
                    plt.grid(True)
                    st.pyplot(plt)
                elif selected_chart_type == "Pie chart":
                    plt.figure(figsize=(10, 6))
                    df[selected_variable].value_counts().plot.pie(autopct='%1.1f%%')
                    plt.title('Pie chart')
                    plt.ylabel('')
                    st.pyplot(plt)
                else:
                    st.warning("Countplot is only available for categorical variables.")
            else:
                if selected_chart_type == "Histogram":
                    st.warning("Histogram is only available for numerical variables.")
                elif selected_chart_type == "Countplot":
                    plt.figure(figsize=(10, 6))
                    sns.countplot(data=df, x=selected_variable)
                    plt.xlabel(selected_variable)
                    plt.ylabel('Count')
                    plt.title('Count of each category')
                    plt.grid(True)
                    st.pyplot(plt)
                else:
                    st.warning("Line plot, Boxplot, and Violin plot are only available for numerical variables.")
        else:
            st.write("Please select a variable to plot.")


        def predict_multi_variable(user_input2):
            # Lấy dữ liệu từ file CSV đã tải lên
            X = df[[user_input2[0], user_input2[1]]].values  # Independent variable
            y = df[user_input2[2]].values  # Dependent variable

            model = LinearRegression()
            model.fit(X, y)

            diab_diagnosis2 = model.predict(X)

            # Vẽ đường hồi quy
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            return diab_diagnosis2, xx, yy, Z


        def predict_knn_multi_variable(user_input2):
            X = df[[user_input2[0], user_input2[1]]].values  # Biến độc lập
            y = df[user_input2[2]].values  # Biến phụ thuộc

            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X, y)

            diab_diagnosis2 = model.predict(X)

            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            return diab_diagnosis2, xx, yy, Z


        def predict_logistic_multi_variable(user_input2):
            X = df[[user_input2[0], user_input2[1]]].values  # Biến độc lập
            y = df[user_input2[2]].values  # Biến phụ thuộc

            model = LogisticRegression()
            model.fit(X, y)

            diab_diagnosis2 = model.predict(X)

            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            return diab_diagnosis2, xx, yy, Z


        with st.container(border=True):
            st.write("Đa biến")
            Independent2 = st.selectbox("Independent variable", (df.columns), index=None, placeholder="Choose value",
                                        key='Independent2')

            Independent3 = st.selectbox("Independent variable", (df.columns), index=None, placeholder="Choose value",
                                        key='Independent3')

            Dependent2 = st.selectbox("Dependent variable", (df.columns), index=None, placeholder="Choose value",
                                      key='Dependent2')

            MH1 = st.selectbox("Mô hình", ("Linear Regression ", "KNN ", "Logistics regression "), index=None,
                               placeholder="Choose Model", key='MH1')

            if MH1 == "Linear Regression ":
                if st.button('Result'):
                    user_input2 = [Independent2, Independent3, Dependent2]
                    diab_diagnosis2, xx, yy, Z = predict_multi_variable(user_input2)

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], df[user_input2[2]], color='red',
                               label='Actual data')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], diab_diagnosis2, color='green',
                               label='Predicted data')
                    ax.plot_surface(xx, yy, Z, alpha=0.5, cmap='viridis', label='Regression Plane')
                    ax.set_xlabel(user_input2[0])
                    ax.set_ylabel(user_input2[1])
                    ax.set_zlabel(user_input2[2])
                    ax.set_title('Linear Regression ')
                    ax.legend()
                    st.pyplot(fig)

            if MH1 == "KNN":
                if st.button('Kết quả đa biến'):
                    user_input2 = [Independent2, Independent3, Dependent2]
                    diab_diagnosis2, xx, yy, Z = predict_knn_multi_variable(user_input2)

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], df[user_input2[2]], color='blue',
                               label='Actual data')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], diab_diagnosis2, color='red',
                               label='Predicted data')
                    ax.plot_surface(xx, yy, Z, alpha=0.5, cmap='viridis', label='Regression Plane')
                    ax.set_xlabel(user_input2[0])
                    ax.set_ylabel(user_input2[1])
                    ax.set_zlabel(user_input2[2])
                    ax.set_title('KNN Regression')
                    ax.legend()
                    st.pyplot(fig)

            if MH1 == "Logistics regression":
                if st.button('Kết quả đa biến'):
                    user_input2 = [Independent2, Independent3, Dependent2]
                    diab_diagnosis2, xx, yy, Z = predict_logistic_multi_variable(user_input2)

                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], df[user_input2[2]], color='blue',
                               label='Actual data')
                    ax.scatter(df[user_input2[0]], df[user_input2[1]], diab_diagnosis2, color='red',
                               label='Predicted data')
                    ax.plot_surface(xx, yy, Z, alpha=0.5, cmap='viridis', label='Regression Plane')
                    ax.set_xlabel(user_input2[0])
                    ax.set_ylabel(user_input2[1])
                    ax.set_zlabel(user_input2[2])
                    ax.set_title('Linear Regression')
                    ax.legend()
                    st.pyplot(fig)










