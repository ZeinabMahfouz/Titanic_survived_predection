import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

def extract_title(name):
    """Extract title from name"""
    title = name.split(',')[1].split('.')[0].strip()
    
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Ms': 'Miss'
    }
    
    return title_mapping.get(title, 'Rare')

def preprocess_data(df, is_train=True, age_medians=None, fare_medians=None, embarked_mode=None):
    """
    Preprocess Titanic dataset
    """
    
    df = df.copy()  # Important: work on a copy
    
    print(f"\n{'='*80}")
    print(f"Processing {'TRAINING' if is_train else 'TEST'} Data")
    print(f"{'='*80}")
    print(f"Original Shape: {df.shape}")
    
    # Store IDs
    passenger_ids = df['PassengerId'].copy()
    
    # Store target if training
    if is_train and 'Survived' in df.columns:
        y = df['Survived'].copy()
    
    print(f"\n{'='*80}")
    print("STEP 1: FEATURE EXTRACTION")
    print(f"{'='*80}")
    
    # Extract Title
    print("\n1.1 Extracting Title from Name...")
    df['Title'] = df['Name'].apply(extract_title)
    print("Title Distribution:")
    print(df['Title'].value_counts())
    
    # Create Family Size
    print("\n1.2 Creating Family Size feature...")
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    
    # Create Is_Alone
    print("1.3 Creating Is_Alone feature...")
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    print(f"\n{'='*80}")
    print("STEP 2: HANDLING MISSING VALUES")
    print(f"{'='*80}")
    
    print("\nMissing Values Before:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Handle Age - Use Title-based median
    print("\n2.1 Filling Age with median by Title...")
    if is_train:
        age_medians = df.groupby('Title')['Age'].median().to_dict()
        print("Median Age by Title (from training data):")
        for title, median_age in age_medians.items():
            print(f"  {title}: {median_age:.1f} years")
    
    df['Age'] = df.apply(
        lambda row: age_medians[row['Title']] if pd.isna(row['Age']) else row['Age'],
        axis=1
    )
    
    # Handle Embarked
    print("\n2.2 Filling Embarked with mode...")
    if is_train:
        embarked_mode = df['Embarked'].mode()[0]
        print(f"Embarked mode (from training data): {embarked_mode}")
    
    df['Embarked'].fillna(embarked_mode, inplace=True)  # FIX: Added inplace=True
    
    # Handle Fare
    print("\n2.3 Filling Fare with median by Pclass...")
    if is_train:
        fare_medians = df.groupby('Pclass')['Fare'].median().to_dict()
        print("Median Fare by Pclass (from training data):")
        for pclass, median_fare in fare_medians.items():
            print(f"  Class {pclass}: ${median_fare:.2f}")
    
    df['Fare'] = df.apply(
        lambda row: fare_medians[row['Pclass']] if pd.isna(row['Fare']) else row['Fare'],
        axis=1
    )
    
    # Drop Cabin
    print("\n2.4 Dropping Cabin column...")
    df.drop('Cabin', axis=1, inplace=True)  # FIX: Added inplace=True
    
    print("\nMissing Values After:")
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(missing_after[missing_after > 0])
    else:
        print("✓ No missing values!")
    
    print(f"\n{'='*80}")
    print("STEP 3: FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    # Create Age Groups
    print("\n3.1 Creating Age Groups...")
    df['Age_Group'] = pd.cut(df['Age'], 
                              bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teenager', 'Young_Adult', 'Adult', 'Senior'])
    
    # Create Fare Groups
    print("3.2 Creating Fare Groups...")
    df['Fare_Group'] = pd.qcut(df['Fare'], 
                                q=4, 
                                labels=['Low', 'Medium', 'High', 'Very_High'],
                                duplicates='drop')
    
    # Create interaction features
    print("3.3 Creating interaction features...")
    df['Age_Class'] = df['Age'] * df['Pclass']
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
    
    print(f"\n{'='*80}")
    print("STEP 4: ENCODING CATEGORICAL VARIABLES")
    print(f"{'='*80}")
    
    # Label Encoding for Sex
    print("\n4.1 Label Encoding for Sex...")
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    
    # One-Hot Encoding
    print("4.2 One-Hot Encoding for Embarked, Title, Age_Group, Fare_Group...")
    df = pd.get_dummies(df, 
                        columns=['Embarked', 'Title', 'Age_Group', 'Fare_Group'],
                        drop_first=True,
                        prefix=['Embarked', 'Title', 'Age_Group', 'Fare_Group'])
    
    print(f"\n{'='*80}")
    print("STEP 5: DROPPING UNNECESSARY COLUMNS")
    print(f"{'='*80}")
    
    # Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Survived']
    cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    
    print(f"\nFeatures remaining: {df.shape[1]} columns")
    
    print(f"\n{'='*80}")
    print("STEP 6: FEATURE SCALING")
    print(f"{'='*80}")
    
    # Identify numerical columns to scale
    numerical_features = ['Age', 'Fare', 'Family_Size', 'Age_Class', 'Fare_Per_Person']  # FIX: Added missing features
    
    if is_train:
        print("\nFitting and transforming scaler on training data...")
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        print("✓ Scaler fitted and applied")
    else:
        print("\nTransforming test data with training scaler...")
        print("⚠ Note: Use the scaler fitted on training data!")
    
    print(f"\nFinal Shape: {df.shape}")
    print(f"Final Features: {df.columns.tolist()}")
    
    # Validation
    print(f"\n{'='*80}")
    print("DATA VALIDATION")
    print(f"{'='*80}")
    
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"⚠ WARNING: Found non-numeric columns: {non_numeric_cols}")
    else:
        print("✓ All columns are numeric")
    
    if df.isnull().sum().sum() > 0:
        print(f"⚠ WARNING: Still have missing values!")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("✓ No missing values")
    
    # Prepare return values
    if is_train:
        statistics = {
            'age_medians': age_medians,
            'fare_medians': fare_medians,
            'embarked_mode': embarked_mode,
            'feature_names': df.columns.tolist()
        }
        return df, y, passenger_ids, statistics
    else:
        return df, passenger_ids

def main():
    """Main preprocessing pipeline"""
    
    print("="*80)
    print("TITANIC DATASET - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv('train.csv')
    print(f"✓ Training data loaded: {train_df.shape}")
    
    # Process training data
    X_train, y_train, train_ids, statistics = preprocess_data(train_df, is_train=True)
    
    # Save training data
    print("\n" + "="*80)
    print("SAVING TRAINING DATA")
    print("="*80)
    X_train.to_csv('X_train_processed.csv', index=False)
    y_train.to_csv('y_train.csv', index=False, header=['Survived'])
    print("✓ X_train_processed.csv saved")
    print("✓ y_train.csv saved")
    
    # Save statistics for test preprocessing
    with open('preprocessing_statistics.pkl', 'wb') as f:
        pickle.dump(statistics, f)
    print("✓ preprocessing_statistics.pkl saved")
    
    # Load test data
    print("\n" + "="*80)
    print("\nLoading test data...")
    test_df = pd.read_csv('test.csv')
    print(f"✓ Test data loaded: {test_df.shape}")
    
    # Process test data using training statistics
    X_test, test_ids = preprocess_data(
        test_df, 
        is_train=False,
        age_medians=statistics['age_medians'],
        fare_medians=statistics['fare_medians'],
        embarked_mode=statistics['embarked_mode']
    )
    
    # Align test columns with training columns
    print("\n" + "="*80)
    print("ALIGNING TEST FEATURES WITH TRAINING FEATURES")
    print("="*80)
    
    # Get missing columns in test
    missing_cols = set(statistics['feature_names']) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
        print(f"Added missing column: {col}")
    
    # Remove extra columns in test
    extra_cols = set(X_test.columns) - set(statistics['feature_names'])
    if extra_cols:
        X_test.drop(list(extra_cols), axis=1, inplace=True)  # FIX: Removed assignment
        print(f"Removed extra columns: {extra_cols}")
    
    # Reorder columns to match training
    X_test = X_test[statistics['feature_names']]
    print(f"✓ Test features aligned: {X_test.shape}")
    
    # Scale test data using training scaler
    print("\nApplying StandardScaler to test data...")
    scaler = StandardScaler()
    # We need to refit on training data to scale test
    numerical_features = ['Age', 'Fare', 'Family_Size', 'Age_Class', 'Fare_Per_Person']  # FIX: Added missing features
    scaler.fit(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    print("✓ Test data scaled")
    
    # Save scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ scaler.pkl saved")
    
    # Save test data
    print("\n" + "="*80)
    print("SAVING TEST DATA")
    print("="*80)
    X_test.to_csv('X_test_processed.csv', index=False)
    test_ids.to_csv('test_ids.csv', index=False, header=['PassengerId'])
    print("✓ X_test_processed.csv saved")
    print("✓ test_ids.csv saved")
    
    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE! ✓")
    print("="*80)
    
    print(f"""
    Training Data: {X_train.shape}
    Test Data: {X_test.shape}
    Files saved successfully!
    """)

if __name__ == "__main__":
    main()