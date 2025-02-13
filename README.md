Blood Type Prediction using Linear Regression  

Project Overview  
This project applies **Linear Regression** to predict a person's **blood type** based on **gender** and **age** using a healthcare dataset. While blood type is categorical data and not typically suited for linear regression, this project serves as an experiment to analyze the effectiveness of regression techniques in classification problems.  

Dataset  
- The dataset consists of **age, gender, and blood type** as features.
- Blood type categories: **A, B, AB, O**.
- Data is collected from a healthcare dataset (source unspecified for privacy reasons).  

Methodology  
1. **Data Preprocessing**:
   - Load the CSV dataset using `pandas`.
   - Encode **gender** (e.g., Male = 0, Female = 1).
   - Convert **blood type** (categorical) into numerical values if needed (e.g., A = 0, B = 1, AB = 2, O = 3).
   - Normalize age values if necessary.
  
2. **Applying Linear Regression**:
   - Use **Scikit-Learn’s Linear Regression** model to predict blood type.
   - Evaluate performance using **Mean Squared Error (MSE)** and **R² Score**.

3. **Limitations**:
   - Blood type is categorical, making **classification models (e.g., Logistic Regression, Decision Trees, or Neural Networks)** more suitable than Linear Regression.
   - The results may not be accurate due to the inappropriate use of regression for categorical prediction.
  
## 🏗️ Installation & Usage  
### **1️⃣ Install Dependencies**  
Ensure you have Python and the required libraries installed:  

first:
pip install pandas numpy scikit-learn matplotlib seaborn

2-Run the Script

Clone this repository and run the script:

git clone https://github.com/Motahhareh-Jafari/Blood-Type-Prediction-using-Linear-Regression
cd Blood-Type-Prediction-using-Linear-Regression
python predict.py

3-Expected Output

The script prints the predicted vs actual blood types along with error metrics.
📊 Results & Discussion

    The model attempts to fit a regression line to a categorical target, which is not ideal.
    Alternative Approach: Consider using Logistic Regression or Decision Trees for classification instead.

Future Improvements
    We can Replace Linear Regression with Logistic Regression or a Neural Network classifier.
    Expand dataset with additional features like ethnicity, medical history, or genetic markers.
    Perform hyperparameter tuning and feature engineering.

📜 License

This project is open-source under the MIT License.
Contributions

Feel free to contribute by opening issues or submitting pull requests!
📩 Contact
GitHub: @Motahhareh-Jafari
