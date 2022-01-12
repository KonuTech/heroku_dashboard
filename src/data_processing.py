from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def make_full_pipeline(df):

    X = df.drop(['Response'], axis = 1)
    y = df.Response.apply(lambda X : 0 if X == 'No' else 1)

    cats = [var for var, var_type in X.dtypes.items() if var_type=='object']
    nums = [var for var in X.columns if var not in cats]

    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ('one_hot_encoder', OneHotEncoder(sparse=False))])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer(
        transformers = [('num_pipeline', num_pipeline, nums),
                        ('cat_pipeline', cat_pipeline, cats)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    _ = full_pipeline.fit_transform(X_train)

    return full_pipeline