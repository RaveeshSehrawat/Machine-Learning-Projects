# 📊 Customer Churn Prediction Dashboard

A machine learning-powered web application for predicting customer churn using Streamlit. This interactive dashboard allows users to input customer information and get real-time predictions about whether a customer is likely to churn or stay.

## 🌟 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard with tabbed navigation
- **Real-time Predictions**: Instant churn probability calculations
- **Comprehensive Analysis**: Detailed risk assessment and recommendations
- **Visual Results**: Progress bars and color-coded risk indicators
- **Professional UI**: Modern design with intuitive navigation
- **35 Feature Support**: Complete customer profile analysis

## 📁 Project Structure

```
Customer Churn Prediction/
├── app.py                                          # Main Streamlit application
├── model.pk1                                       # Trained machine learning model
├── Finding_best_model_for_Customer_churn_prediciton.ipynb  # Model training notebook
├── requirements.txt                                # Python dependencies
├── README.md                                       # Project documentation
└── venv/                                          # Virtual environment (created during setup)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Windows PowerShell (for Windows users)

### Installation

1. **Clone or download the project**
   ```bash
   # Navigate to the project directory
   cd "d:\Downloads\Customer Churn Prediction"
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit app**
   ```powershell
   & "D:/Downloads/Customer Churn Prediction/venv/Scripts/python.exe" -m streamlit run app.py
   ```

2. **Access the dashboard**
   - Local URL: http://localhost:8501
   - Network URL: http://192.168.x.x:8501
   - The URLs will be displayed in the terminal

## 📊 Dashboard Overview

### Navigation Tabs

#### 👤 Customer Info
- **Basic Information**: Senior citizen status, tenure, charges
- **Demographics**: Gender, partner, dependents

#### 📞 Phone Services  
- Phone service availability
- Multiple lines configuration

#### 🌐 Internet Services
- Internet service type (DSL, Fiber optic, No service)
- Add-on services: Security, backup, protection, tech support
- Streaming services: TV and movies

#### 💳 Billing
- Contract terms (Month-to-month, One year, Two year)
- Billing preferences and payment methods

### Prediction Results

The dashboard provides comprehensive analysis including:

- **🎯 Primary Prediction**: Churn/Stay classification
- **📈 Probability Scores**: Numerical likelihood percentages
- **⚠️ Risk Assessment**: High/Medium/Low risk categorization
- **💡 Recommendations**: Actionable insights
- **📊 Visual Indicators**: Progress bars and color-coded results
- **📋 Customer Summary**: Key customer information overview

## 🔧 Technical Details

### Model Features (35 total)

The model uses the following features for prediction:

**Numerical Features:**
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `tenure`: Number of months with the company
- `MonthlyCharges`: Monthly service charges
- `TotalCharges`: Total charges to date

**Categorical Features (One-Hot Encoded):**
- Demographics: Gender, Partner, Dependents
- Phone Services: Phone service, Multiple lines
- Internet Services: Service type and add-ons
- Contract: Contract length and billing preferences
- Payment: Payment method

### Dependencies

```
streamlit>=1.50.0
numpy>=2.3.3
pandas>=2.3.3
scikit-learn
pickle (built-in Python library)
```

## 🎨 UI Features

- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎭 Professional Styling**: Modern interface with consistent theming
- **💡 Interactive Help**: Tooltips and guidance throughout
- **📊 Visual Analytics**: Charts and progress indicators
- **🚀 Real-time Updates**: Instant prediction results

## 📈 Usage Guide

### Step-by-Step Instructions

1. **Launch the application** using the installation steps above
2. **Navigate through tabs** to enter customer information:
   - Start with Customer Info tab for basic details
   - Move to Phone Services for phone-related features
   - Configure Internet Services and add-ons
   - Set up Billing and contract information

3. **Make predictions** by clicking the "🔮 Predict Customer Churn" button
4. **Analyze results** in the comprehensive results section:
   - Review primary prediction (Churn/Stay)
   - Check probability percentages
   - Note risk level and recommendations
   - Examine visual probability indicators

### Interpretation of Results

- **🟢 Low Risk (0-40% churn probability)**: Customer likely to stay
- **🟡 Medium Risk (40-70% churn probability)**: Monitor customer closely
- **🔴 High Risk (70-100% churn probability)**: Immediate retention action needed

## 🔬 Model Information

The prediction model is a trained machine learning classifier that:
- Processes 35 customer features
- Provides binary classification (Churn/No Churn)
- Returns probability scores for both outcomes
- Has been trained on historical customer data

## 🚨 Troubleshooting

### Common Issues

**Streamlit not found:**
```powershell
# Use the full Python path
& "D:/Downloads/Customer Churn Prediction/venv/Scripts/python.exe" -m streamlit run app.py
```

**Module import errors:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1
# Reinstall dependencies
pip install -r requirements.txt
```

**Model file not found:**
- Ensure `model.pk1` is in the project directory
- Check file permissions and accessibility

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify the virtual environment is activated
3. Ensure all required files are present
4. Check Python version compatibility (3.7+)

## 📝 Development

### Adding New Features

To extend the dashboard:
1. Modify `app.py` for UI changes
2. Update model handling for new features
3. Test thoroughly with various input combinations

### Model Updates

To use a different model:
1. Replace `model.pk1` with your trained model
2. Ensure feature compatibility (35 features expected)
3. Test prediction functionality

## 📄 License

This project is for educational and business use. Please ensure compliance with your organization's data handling policies when using customer data.

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📞 Support

For questions, issues, or feature requests, please contact your data science team or system administrator.

**Happy Predicting! 🎉**
