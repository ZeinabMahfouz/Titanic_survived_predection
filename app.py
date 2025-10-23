import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BEAUTIFUL STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    
    .sub-title {
        font-size: 1.5em;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction result */
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin: 20px 0;
        animation: fadeIn 0.5s;
    }
    
    .survived {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .not-survived {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'final_model.pkl' not found!")
        st.info("Please train your model first and save it using: `pickle.dump(final_model, open('final_model.pkl', 'wb'))`")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    """Load the fitted scaler"""
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except:
        st.warning("⚠️ Scaler not found. Creating new scaler.")
        return StandardScaler()

@st.cache_resource
def load_preprocessing_stats():
    """Load preprocessing statistics"""
    try:
        with open('preprocessing_statistics.pkl', 'rb') as f:
            stats = pickle.load(f)
        return stats
    except:
        st.warning("⚠️ Preprocessing statistics not found.")
        return None

@st.cache_data
def load_data():
    """Load training data for visualizations"""
    try:
        X_train = pd.read_csv('X_train_processed.csv')
        y_train = pd.read_csv('y_train.csv')
        train_raw = pd.read_csv('train.csv')
        return X_train, y_train, train_raw
    except:
        return None, None, None

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    try:
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return features
    except:
        # Load from processed training data as fallback
        try:
            X_train = pd.read_csv('X_train_processed.csv')
            return X_train.columns.tolist()
        except:
            return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_title(name):
    """Extract title from name"""
    try:
        title = name.split(',')[1].split('.')[0].strip()
    except:
        title = 'Mr'
    
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Ms': 'Miss'
    }
    
    return title_mapping.get(title, 'Rare')

def preprocess_input(data, preprocessing_stats, scaler, feature_names):
    """
    Preprocess user input to match training data format
    """
    # Extract title
    title = extract_title(data['Name'])
    
    # Create family size
    family_size = data['SibSp'] + data['Parch'] + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Fill missing age using preprocessing stats if available
    age = data['Age']
    if preprocessing_stats and 'age_medians' in preprocessing_stats:
        if pd.isna(age):
            age = preprocessing_stats['age_medians'].get(title, 30)
    
    # Create interaction features (before scaling)
    age_class = age * data['Pclass']
    fare_per_person = data['Fare'] / family_size
    
    # Encode Sex
    sex_encoded = 0 if data['Sex'].lower() == 'female' else 1
    
    # Create base feature dictionary with numeric values only
    features_dict = {
        'Pclass': data['Pclass'],
        'Sex': sex_encoded,
        'Age': age,
        'SibSp': data['SibSp'],
        'Parch': data['Parch'],
        'Fare': data['Fare'],
        'Family_Size': family_size,
        'Is_Alone': is_alone,
        'Age_Class': age_class,
        'Fare_Per_Person': fare_per_person
    }
    
    # Add one-hot encoded features (all set to 0 initially)
    # Embarked
    features_dict['Embarked_Q'] = 0
    features_dict['Embarked_S'] = 0
    if data['Embarked'] == 'Q':
        features_dict['Embarked_Q'] = 1
    elif data['Embarked'] == 'S':
        features_dict['Embarked_S'] = 1
    
    # Title
    features_dict['Title_Miss'] = 1 if title == 'Miss' else 0
    features_dict['Title_Mr'] = 1 if title == 'Mr' else 0
    features_dict['Title_Mrs'] = 1 if title == 'Mrs' else 0
    features_dict['Title_Rare'] = 1 if title == 'Rare' else 0
    
    # Age Group
    if age <= 12:
        age_group = 'Child'
    elif age <= 18:
        age_group = 'Teenager'
    elif age <= 35:
        age_group = 'Young_Adult'
    elif age <= 60:
        age_group = 'Adult'
    else:
        age_group = 'Senior'
    
    # Age group one-hot (drop first, so we don't include 'Child')
    features_dict['Age_Group_Teenager'] = 1 if age_group == 'Teenager' else 0
    features_dict['Age_Group_Young_Adult'] = 1 if age_group == 'Young_Adult' else 0
    features_dict['Age_Group_Adult'] = 1 if age_group == 'Adult' else 0
    features_dict['Age_Group_Senior'] = 1 if age_group == 'Senior' else 0
    
    # Fare Group (quartiles based on training data)
    if data['Fare'] <= 7.91:
        fare_group = 'Low'
    elif data['Fare'] <= 14.454:
        fare_group = 'Medium'
    elif data['Fare'] <= 31:
        fare_group = 'High'
    else:
        fare_group = 'Very_High'
    
    # Fare group one-hot (drop first, so we don't include 'Low')
    features_dict['Fare_Group_Medium'] = 1 if fare_group == 'Medium' else 0
    features_dict['Fare_Group_High'] = 1 if fare_group == 'High' else 0
    features_dict['Fare_Group_Very_High'] = 1 if fare_group == 'Very_High' else 0
    
    # Create DataFrame with correct column order
    if feature_names:
        # Initialize all features to 0
        features_aligned = {col: 0 for col in feature_names}
        # Update with our calculated features
        for key, value in features_dict.items():
            if key in features_aligned:
                features_aligned[key] = value
        
        df = pd.DataFrame([features_aligned])
    else:
        df = pd.DataFrame([features_dict])
    
    # Scale numerical features
    numerical_features = ['Age', 'Fare', 'Family_Size', 'Age_Class', 'Fare_Per_Person']
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    if scaler and len(numerical_features) > 0:
        df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.markdown('<h1 class="main-title">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict survival on the Titanic using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    scaler = load_scaler()
    preprocessing_stats = load_preprocessing_stats()
    feature_names = load_feature_names()
    X_train, y_train, train_raw = load_data()
    
    # Check if model is loaded
    model_loaded = model is not None
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please train and save your model first!")
        st.info("""
        **To use this app with your model:**
        1. Train your model using `model_evaluation.py`
        2. Save it: `pickle.dump(final_model, open('final_model.pkl', 'wb'))`
        3. Restart this app
        """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: PREDICTION
    # ========================================================================
    with tab1:
        st.markdown("### Make a Prediction")
        
        if not model_loaded:
            st.warning("⚠️ Model not loaded. Prediction unavailable.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 👤 Personal Information")
            
            name = st.text_input("Full Name", "Doe, Mr. John", help="Enter passenger's full name (Last, Title. First)")
            
            sex = st.selectbox("Gender", ["Female", "Male"], help="Select gender")
            
            age = st.slider("Age", 0, 80, 30, help="Select passenger's age")
            
            pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                                 help="1 = First Class, 2 = Second Class, 3 = Third Class")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 🎫 Travel Information")
            
            sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0,
                                   help="Number of siblings or spouses aboard")
            
            parch = st.number_input("Number of Parents/Children", 0, 6, 0,
                                   help="Number of parents or children aboard")
            
            fare = st.number_input("Fare (in £)", 0.0, 500.0, 32.0,
                                  help="Ticket fare in British Pounds")
            
            embarked = st.selectbox("Port of Embarkation", 
                                   ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
                                   help="Port where passenger boarded")
            embarked = embarked[embarked.find("(")+1:embarked.find(")")]
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            predict_button = st.button("🔮 Predict Survival", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            input_data = {
                'Name': name,
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked
            }
            
            # Preprocess and predict
            try:
                # Preprocess input
                features_df = preprocess_input(input_data, preprocessing_stats, scaler, feature_names)
                
                # Make prediction
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0]
                
                # Display result
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-box survived">
                            ✅ SURVIVED<br>
                            <span style="font-size: 0.6em;">Survival Probability: {probability[1]*100:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                        <div class="prediction-box not-survived">
                            ❌ DID NOT SURVIVE<br>
                            <span style="font-size: 0.6em;">Survival Probability: {probability[1]*100:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎫 Class</h3>
                            <h2>{pclass}</h2>
                            <p>{"First" if pclass==1 else "Second" if pclass==2 else "Third"} Class</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>👨‍👩‍👧‍👦 Family</h3>
                            <h2>{sibsp + parch + 1}</h2>
                            <p>Members aboard</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>💰 Fare</h3>
                            <h2>£{fare:.2f}</h2>
                            <p>Ticket price</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    confidence = max(probability) * 100
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎯 Confidence</h3>
                            <h2>{confidence:.1f}%</h2>
                            <p>Model certainty</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show feature values (expandable)
                with st.expander("🔍 View Feature Values"):
                    st.dataframe(features_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.error("Please check that all preprocessing files are available.")
                import traceback
                st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 2: DATA EXPLORER
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Explore the Titanic Dataset")
        
        if train_raw is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>👥 Total Passengers</h3>
                        <h2>{len(train_raw)}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                survived = train_raw['Survived'].sum()
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>✅ Survived</h3>
                        <h2>{survived}</h2>
                        <p>{survived/len(train_raw)*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                died = len(train_raw) - survived
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>❌ Died</h3>
                        <h2>{died}</h2>
                        <p>{died/len(train_raw)*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_age = train_raw['Age'].mean()
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>📅 Avg Age</h3>
                        <h2>{avg_age:.1f}</h2>
                        <p>years old</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Survival by Gender
                fig1 = px.bar(train_raw.groupby(['Sex', 'Survived']).size().reset_index(name='Count'),
                             x='Sex', y='Count', color='Survived',
                             title='Survival by Gender',
                             color_discrete_map={0: '#ee0979', 1: '#11998e'},
                             labels={'Survived': 'Status', 'Count': 'Number of Passengers'},
                             barmode='group')
                fig1.update_layout(height=400, plot_bgcolor='white')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Survival by Class
                fig2 = px.bar(train_raw.groupby(['Pclass', 'Survived']).size().reset_index(name='Count'),
                             x='Pclass', y='Count', color='Survived',
                             title='Survival by Passenger Class',
                             color_discrete_map={0: '#ee0979', 1: '#11998e'},
                             labels={'Survived': 'Status', 'Count': 'Number of Passengers'},
                             barmode='group')
                fig2.update_layout(height=400, plot_bgcolor='white')
                st.plotly_chart(fig2, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Age Distribution
                fig3 = px.histogram(train_raw, x='Age', color='Survived',
                                   title='Age Distribution by Survival',
                                   color_discrete_map={0: '#ee0979', 1: '#11998e'},
                                   labels={'Survived': 'Status'},
                                   nbins=30)
                fig3.update_layout(height=400, plot_bgcolor='white')
                st.plotly_chart(fig3, use_container_width=True)
            
            with col4:
                # Fare Distribution
                fig4 = px.box(train_raw, x='Survived', y='Fare',
                             title='Fare Distribution by Survival',
                             color='Survived',
                             color_discrete_map={0: '#ee0979', 1: '#11998e'},
                             labels={'Survived': 'Status'})
                fig4.update_layout(height=400, plot_bgcolor='white')
                st.plotly_chart(fig4, use_container_width=True)
            
            # Data table
            st.markdown("#### 📋 Sample Data")
            st.dataframe(train_raw.head(10), use_container_width=True)
        
        else:
            st.warning("⚠️ Training data not found. Please ensure 'train.csv' is in the directory.")
    
    # ========================================================================
    # TAB 3: MODEL PERFORMANCE
    # ========================================================================
    with tab3:
        st.markdown("### 📈 Model Performance Metrics")
        
        if X_train is not None and y_train is not None and model_loaded:
            # Calculate actual metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(X_train)
            
            accuracy = accuracy_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Accuracy</h3>
                        <h2>{accuracy*100:.1f}%</h2>
                        <p>Overall correctness</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Precision</h3>
                        <h2>{precision*100:.1f}%</h2>
                        <p>True positive rate</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔍 Recall</h3>
                        <h2>{recall*100:.1f}%</h2>
                        <p>Sensitivity</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚖️ F1-Score</h3>
                        <h2>{f1*100:.1f}%</h2>
                        <p>Harmonic mean</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Feature importance
            if hasattr(model, 'coef_'):
                st.markdown("#### 🔑 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': np.abs(model.coef_[0])
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_importance = go.Figure(go.Bar(
                    x=feature_importance['Importance'],
                    y=feature_importance['Feature'],
                    orientation='h',
                    marker=dict(
                        color=feature_importance['Importance'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                fig_importance.update_layout(
                    title='Top 15 Most Important Features',
                    xaxis_title='Importance Score (Absolute Coefficient)',
                    yaxis_title='Features',
                    height=500,
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("#### 📊 Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_train, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: Died', 'Predicted: Survived'],
                y=['Actual: Died', 'Actual: Survived'],
                colorscale='RdYlGn',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                showscale=True
            ))
            fig_cm.update_layout(
                title='Confusion Matrix - Training Data',
                height=400,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        else:
            st.warning("⚠️ Model or training data not available for performance metrics.")
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    with tab4:
        st.markdown("### ℹ️ About This Application")
        
        st.markdown("""
            <div class="info-box">
                <h3>🚢 The Titanic Dataset</h3>
                <p>
                The sinking of the Titanic is one of the most infamous shipwrecks in history. 
                On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" 
                RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't 
                enough lifeboats for everyone onboard, resulting in the death of 1502 out of 
                2224 passengers and crew.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3>🤖 Machine Learning Model</h3>
                <p>
                This application uses a <strong>Logistic Regression</strong> model trained on 
                historical Titanic passenger data. The model analyzes various features such as:
                </p>
                <ul>
                    <li>👤 Gender and Age</li>
                    <li>🎫 Passenger Class</li>
                    <li>👨‍👩‍👧‍👦 Family Size</li>
                    <li>💰 Ticket Fare</li>
                    <li>📍 Port of Embarkation</li>
                    <li>🏷️ Title (Mr, Mrs, Miss, etc.)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3>📊 How It Works</h3>
                <ol>
                    <li><strong>Data Preprocessing:</strong> Your input is cleaned and transformed</li>
                    <li><strong>Feature Engineering:</strong> Additional features are created (family size, titles, etc.)</li>
                    <li><strong>Encoding:</strong> Categorical variables are converted to numbers</li>
                    <li><strong>Scaling:</strong> Numerical features are standardized</li>
                    <li><strong>Prediction:</strong> The model calculates survival probability</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        if model_loaded:
            st.markdown(f"""
                <div class="info-box">
                    <h3>✅ Model Status</h3>
                    <p><strong>Model:</strong> Loaded Successfully</p>
                    <p><strong>Type:</strong> Logistic Regression</p>
                    <p><strong>Features:</strong> {len(feature_names) if feature_names else 'Unknown'}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-box">
                    <h3>⚠️ Model Status</h3>
                    <p><strong>Status:</strong> Not Loaded</p>
                    <p>Please train and save your model to use predictions.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3>🛠️ Technologies Used</h3>
                <ul>
                    <li><strong>Streamlit</strong> - Web application framework</li>
                    <li><strong>Scikit-learn</strong> - Machine learning library</li>
                    <li><strong>Pandas</strong> - Data manipulation</li>
                    <li><strong>Plotly</strong> - Interactive visualizations</li>
                    <li><strong>NumPy</strong> - Numerical computing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3>📝 How to Use</h3>
                <ol>
                    <li>Go to the <strong>Prediction</strong> tab</li>
                    <li>Enter passenger information</li>
                    <li>Click <strong>Predict Survival</strong></li>
                    <li>View the prediction and probability</li>
                    <li>Explore data insights in other tabs</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3>📚 Key Insights from Historical Data</h3>
                <ul>
                    <li>👩 <strong>Women</strong> had a 74% survival rate vs 19% for men</li>
                    <li>🎫 <strong>First-class</strong> passengers had 63% survival vs 24% in third class</li>
                    <li>👶 <strong>Children</strong> under 10 had a 61% survival rate</li>
                    <li>👨‍👩‍👧‍👦 Small families (2-4 members) had better survival chances</li>
                    <li>💰 Higher fare generally correlated with better survival</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()