import pandas as pd
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

pd.set_option('display.max_columns', None)

TARGET = 'health_condition'


def df_mask(df):
    df['diet_type'] = df['diet_type'].replace({'veg': 2, 'balanced': 1, 'non-veg': 0})
    df['stress_level'] = df['stress_level'].replace({'high': 2, 'medium': 1, 'low': 0})
    df['sleep_quality'] = df['sleep_quality'].replace({'good': 2, 'average': 1, 'poor': 0})
    df['physical_activity_level'] = df['physical_activity_level'].replace({'active': 12, 'moderate': 1, 'sedentary': 0})
    df['smoking_alcohol'] = df['smoking_alcohol'].replace({'yes': 2, 'occasional': 1, 'no': 0})
    df['gender'] = df['gender'].replace({'male': 2, 'female': 1, 'other': 0})
    return df


def train_encode(df):
    df[TARGET] = df[TARGET].replace({'fit': 2, 'at-risk': 1, 'unhealthy': 0})
    return df


def remove_na(df, stats=None):
    if stats is None:
        stats = {
            'group_means': {
                ('step_count', 'physical_activity_level'):
                    df.groupby('physical_activity_level')['step_count'].mean(),
                ('exercise_duration', 'physical_activity_level'):
                    df.groupby('physical_activity_level')['exercise_duration'].mean(),
                ('calorie_expenditure', 'physical_activity_level'):
                    df.groupby('physical_activity_level')['calorie_expenditure'].mean(),
                ('sleep_duration', 'sleep_quality'):
                    df.groupby('sleep_quality')['sleep_duration'].mean(),
            },
            'medians': df.median(numeric_only=True),
            'gender_mode': df['gender'].mode()[0],
        }

    for (col, by), mapping in stats['group_means'].items():
        df[col] = df[col].fillna(df[by].map(mapping))

    df['gender'] = df['gender'].fillna(stats['gender_mode'])

    for col in df.columns:
        if col in ('id', TARGET):
            continue
        if col in stats['medians'].index:
            df[col] = df[col].fillna(stats['medians'][col])

    return df, stats


def model_lgb(df):
    X = df.drop(columns=['id', TARGET])
    y = df[TARGET]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    lgb = LGBMClassifier(
        n_estimators=5000,       
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1,
    )

    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=100),
            lightgbm.log_evaluation(period=100),
        ],
    )

    y_val_pred = lgb.predict(X_val)
    print(balanced_accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))
    return lgb


def predict(df_test, model, train_columns):
    X_test = df_test[train_columns]
    return model.predict(X_test)


def decode(df_test, preds):
    inverse_mapping = {2: 'fit', 1: 'at-risk', 0: 'unhealthy'}
    y_pred_text = pd.Series(preds).map(inverse_mapping)

    submission = pd.DataFrame({
        'id': df_test['id'],
        TARGET: y_pred_text,
    })
    submission.to_csv('kaggle/Predicting Student Health Risk/submission.csv', index=False)
    print('submission.csv saved')


if __name__ == '__main__':
    df_train = pd.read_csv('kaggle/Predicting Student Health Risk/train.csv')
    train_encode(df_train)
    df_mask(df_train)
    df_train, stats = remove_na(df_train)          

    df_test = pd.read_csv('kaggle/Predicting Student Health Risk/test.csv')
    df_mask(df_test)
    df_test, _ = remove_na(df_test, stats)
    
    lgb = model_lgb(df_train)

    train_columns = df_train.drop(columns=['id', TARGET]).columns
    preds = predict(df_test, lgb, train_columns)
    decode(df_test, preds)
