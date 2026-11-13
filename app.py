import json
import time
from datetime import datetime
from io import StringIO

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


from src.config import DATA_DIR, FEATURE_ENGINEERING, MODELS, RESULTS_DIR
from src.utils.feature_engineering import get_feature_engineering
from src.utils.model_training import create_pipeline, tune_hyperparameters


st.set_page_config(
    page_title="Malicious Traffic Detection",
    page_icon="🛡️",
    layout="wide",
)


def read_dataset(uploaded_file):
    """Read a CSV file or a ZIP archive containing one CSV file."""
    uploaded_file.seek(0)
    compression = "zip" if uploaded_file.name.lower().endswith(".zip") else None
    return pd.read_csv(uploaded_file, compression=compression)


def prepare_features(
    data: pd.DataFrame,
    target_column: str,
    expected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate labels and return numeric, finite features in a stable order."""
    if target_column not in data.columns:
        raise ValueError(f"The dataset is missing the target column '{target_column}'.")

    labels: pd.Series = pd.Series(data.loc[:, target_column])
    if labels.isna().any():
        raise ValueError(f"The target column '{target_column}' contains missing labels.")

    features = data.drop(columns=[target_column]).copy()
    if expected_columns is not None:
        missing = [column for column in expected_columns if column not in features.columns]
        extra = [column for column in features.columns if column not in expected_columns]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(map(str, missing))}")
            if extra:
                details.append(f"unexpected columns: {', '.join(map(str, extra))}")
            raise ValueError("Train/test feature columns do not match (" + "; ".join(details) + ").")
        features = features.reindex(columns=expected_columns)

    for column in features.columns:
        source_values: pd.Series = pd.Series(features.loc[:, column])
        converted = pd.Series(
            pd.to_numeric(source_values, errors="coerce"), index=source_values.index
        )
        invalid = source_values.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Feature '{column}' contains non-numeric values. "
                "This pipeline requires numeric traffic features."
            )
        features.loc[:, column] = converted

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    if features.empty or features.shape[1] == 0:
        raise ValueError("The dataset must include at least one feature column.")
    return features, labels


def prepare_unlabeled_features(data: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Validate and align unlabeled input to the saved model's feature schema."""
    if data.columns.has_duplicates:
        raise ValueError("The input file contains duplicate column names.")

    missing = [column for column in expected_columns if column not in data.columns]
    extra = [column for column in data.columns if column not in expected_columns]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(map(str, missing))}")
        if extra:
            details.append(f"unexpected columns: {', '.join(map(str, extra))}")
        raise ValueError("Input features do not match the model (" + "; ".join(details) + ").")

    features = data.reindex(columns=expected_columns).copy()
    for column in expected_columns:
        source_values = pd.Series(features.loc[:, column])
        converted = pd.Series(
            pd.to_numeric(source_values, errors="coerce"), index=source_values.index
        )
        invalid = source_values.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Feature '{column}' contains non-numeric values.")
        features.loc[:, column] = converted

    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def render_prediction_workflow():
    """Classify unlabeled flow rows using a locally saved trusted model."""
    st.subheader("Predict on new flows")
    st.write(
        "Upload a CSV or ZIP containing the same numeric feature columns used to train "
        "a saved model. Do not include the target/label column."
    )
    st.warning(
        "Only load model artifacts created by this project that you trust. "
        "Joblib model files can execute code when loaded."
    )

    model_paths = []
    if RESULTS_DIR.exists():
        model_paths = sorted(
            [
                *RESULTS_DIR.glob("ml_data_*_model.joblib"),
                *RESULTS_DIR.glob("ml_data_*_model.pkl"),
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not model_paths:
        st.info("No saved model was found in results/. Train and save a model first.")
        return

    selected_model_name = st.selectbox(
        "Saved model", options=[path.name for path in model_paths], key="inference_model"
    )
    selected_model_path = next(path for path in model_paths if path.name == selected_model_name)
    uploaded_file = st.file_uploader(
        "Unlabeled flow data (CSV or ZIP containing a CSV)",
        type=["csv", "zip"],
        key="inference_upload",
    )

    if uploaded_file is not None:
        try:
            preview = read_dataset(uploaded_file)
            st.caption(f"Loaded {len(preview):,} flow rows and {len(preview.columns)} columns.")
            st.dataframe(preview.head(10), use_container_width=True)
        except Exception as error:
            st.error(f"Could not read the uploaded data: {error}")
            return

    if st.button("Classify flows", type="primary", key="classify_flows"):
        if uploaded_file is None:
            st.error("Upload a CSV or ZIP file before classifying flows.")
            return

        try:
            input_data = read_dataset(uploaded_file)
            pipeline = joblib.load(selected_model_path)
            expected_features = getattr(pipeline, "feature_names_in_", None)
            if expected_features is None:
                raise ValueError(
                    "This model does not contain its original feature names. "
                    "Retrain and save it with the Streamlit app before using inference."
                )

            features = prepare_unlabeled_features(input_data, list(expected_features))
            predictions = pipeline.predict(features)
            output = features.copy()
            output["prediction"] = predictions

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(features)
                for class_index, class_label in enumerate(pipeline.classes_):
                    output[f"probability_{class_label}"] = probabilities[:, class_index]

            summary = (
                pd.Series(predictions, name="Predicted class")
                .astype(str)
                .value_counts()
                .rename_axis("Predicted class")
                .reset_index(name="Flows")
            )
            st.success(f"Classified {len(output):,} flows with {selected_model_name}.")
            st.bar_chart(summary.set_index("Predicted class"))
            st.dataframe(output.head(500), use_container_width=True)

            prediction_csv = output.to_csv(index=False).encode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}.csv"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / output_filename).write_bytes(prediction_csv)
            st.download_button(
                "Download predictions",
                data=prediction_csv,
                file_name=output_filename,
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Could not classify uploaded flows: {error}")


def render_data_visualizations(data, target_column):
    """Show class balance, a feature distribution, and feature correlations."""
    st.subheader("Data visualizations")
    class_column, feature_column = st.columns(2)

    class_counts = (
        data[target_column]
        .astype("string")
        .fillna("(missing)")
        .astype(str)
        .value_counts()
        .rename_axis("Traffic class")
        .reset_index(name="Flows")
    )
    with class_column:
        st.caption("Traffic class balance")
        balance_chart = (
            alt.Chart(class_counts)
            .mark_bar()
            .encode(
                x=alt.X("Traffic class:N", sort="-y", title="Class"),
                y=alt.Y("Flows:Q", title="Number of flows"),
                color=alt.Color("Traffic class:N", legend=None),
                tooltip=["Traffic class:N", "Flows:Q"],
            )
        )
        st.altair_chart(balance_chart, use_container_width=True)

    numeric_features = [
        column
        for column in data.columns
        if column != target_column
        and pd.to_numeric(data[column], errors="coerce").notna().any()
    ]
    if not numeric_features:
        with feature_column:
            st.info("No numeric feature columns are available to plot.")
        return

    with feature_column:
        selected_feature = st.selectbox(
            "Feature distribution", options=numeric_features, key="distribution_feature"
        )
        distribution_data = data[[selected_feature, target_column]].copy()
        distribution_data = distribution_data.rename(
            columns={selected_feature: "Feature value", target_column: "Traffic class"}
        )
        distribution_data["Feature value"] = pd.to_numeric(
            distribution_data["Feature value"], errors="coerce"
        )
        distribution_data["Traffic class"] = distribution_data["Traffic class"].astype(str)
        distribution_data = distribution_data.dropna(subset=["Feature value"])
        if len(distribution_data) > 10000:
            distribution_data = distribution_data.sample(10000, random_state=42)

        st.caption("Feature values by traffic class")
        distribution_chart = (
            alt.Chart(distribution_data)
            .mark_bar(opacity=0.65)
            .encode(
                x=alt.X("Feature value:Q", bin=alt.Bin(maxbins=30), title=selected_feature),
                y=alt.Y("count():Q", stack=None, title="Number of flows"),
                color=alt.Color("Traffic class:N", title="Class"),
                tooltip=["Traffic class:N", alt.Tooltip("count():Q", title="Flows")],
            )
        )
        st.altair_chart(distribution_chart, use_container_width=True)

    with st.expander("Feature correlation heatmap", expanded=False):
        default_features = numeric_features[: min(12, len(numeric_features))]
        selected_features = st.multiselect(
            "Features to include (maximum 20)",
            options=numeric_features,
            default=default_features,
            max_selections=20,
            key="correlation_features",
        )
        if not selected_features:
            st.info("Select at least one numeric feature to draw the heatmap.")
        else:
            correlation_data = data[selected_features].apply(pd.to_numeric, errors="coerce")
            if len(correlation_data) > 20000:
                correlation_data = correlation_data.sample(20000, random_state=42)
            correlation_data = correlation_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            correlation_long = (
                correlation_data.corr()
                .fillna(0)
                .rename_axis("Feature")
                .reset_index()
                .melt(id_vars="Feature", var_name="Compared feature", value_name="Correlation")
            )
            heatmap = alt.Chart(correlation_long).mark_rect().encode(
                x=alt.X("Compared feature:N", sort=selected_features, title=None),
                y=alt.Y("Feature:N", sort=selected_features, title=None),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True),
                    title="Correlation",
                ),
                tooltip=["Feature:N", "Compared feature:N", alt.Tooltip("Correlation:Q", format=".3f")],
            )
            heatmap_labels = alt.Chart(correlation_long).mark_text(fontSize=10).encode(
                x=alt.X("Compared feature:N", sort=selected_features),
                y=alt.Y("Feature:N", sort=selected_features),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.value("black"),
            )
            st.altair_chart(heatmap + heatmap_labels, use_container_width=True)


def get_model_feature_importance(pipeline, input_features):
    """Return feature importance for estimators that expose it."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_)).mean(axis=0)
    else:
        return None

    transformer = pipeline.named_steps.get("feature_engineering")
    try:
        names = (
            transformer.get_feature_names_out(input_features)
            if transformer is not None
            else input_features
        )
    except (AttributeError, TypeError, ValueError):
        names = [f"Feature {index + 1}" for index in range(len(values))]

    if len(names) != len(values):
        names = [f"Feature {index + 1}" for index in range(len(values))]
    importance = pd.DataFrame({"Feature": names, "Importance": values})
    return importance.sort_values("Importance", ascending=False).head(20)


def render_results(result):
    st.subheader("Evaluation results")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Test accuracy", f"{result['accuracy']:.2%}")
    metric_columns[1].metric("Training rows", f"{result['training_rows']:,}")
    metric_columns[2].metric("Test rows", f"{result['test_rows']:,}")
    metric_columns[3].metric("Training time", f"{result['training_time']:.2f} s")

    if result["cv_scores"] is not None:
        st.metric(
            "Cross-validation accuracy",
            f"{result['cv_scores'].mean():.2%} ± {result['cv_scores'].std():.2%}",
        )

    if result["best_params"]:
        st.write("**Best hyperparameters**")
        st.json(result["best_params"])

    left, right = st.columns(2)
    with left:
        st.write("**Classification report**")
        st.dataframe(result["report_table"], use_container_width=True)
    with right:
        st.write("**Confusion matrix**")
        confusion_long = (
            result["confusion_table"]
            .rename_axis("Actual")
            .reset_index()
            .melt(id_vars="Actual", var_name="Predicted", value_name="Flows")
        )
        confusion_heatmap = (
            alt.Chart(confusion_long)
            .mark_rect()
            .encode(
                x=alt.X("Predicted:N", title=None),
                y=alt.Y("Actual:N", title=None),
                color=alt.Color("Flows:Q", scale=alt.Scale(scheme="blues"), title="Flows"),
                tooltip=["Actual:N", "Predicted:N", "Flows:Q"],
            )
        )
        confusion_labels = alt.Chart(confusion_long).mark_text().encode(
            x=alt.X("Predicted:N"),
            y=alt.Y("Actual:N"),
            text=alt.Text("Flows:Q"),
            color=alt.value("black"),
        )
        st.altair_chart(confusion_heatmap + confusion_labels, use_container_width=True)
        st.dataframe(result["confusion_table"], use_container_width=True)

    if result["feature_importance"] is not None:
        st.write("**Top model feature importance**")
        importance_chart = (
            alt.Chart(result["feature_importance"])
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Importance"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
            )
        )
        st.altair_chart(importance_chart, use_container_width=True)

    st.download_button(
        "Download trained model",
        data=result["model_bytes"],
        file_name=result["model_filename"],
        mime="application/octet-stream",
    )
    st.download_button(
        "Download test predictions",
        data=result["predictions_csv"],
        file_name="traffic_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download evaluation report",
        data=result["report_text"],
        file_name=result["report_filename"],
        mime="text/plain",
    )


st.title("🛡️ Malicious Traffic Detection")
st.caption("Train models on labeled flows or classify new flows with a saved model.")
st.info(
    "Data is processed by this app instance. Training requires a target label; "
    "prediction requires the same numeric feature columns used to train the saved model."
)
workflow = st.radio(
    "Workflow",
    options=["Train and evaluate", "Predict new flows"],
    horizontal=True,
)
algorithm = next(iter(MODELS))
feature_engineering = "None"
tune = False
cross_validation = False

with st.sidebar:
    if workflow == "Train and evaluate":
        st.header("Model configuration")
        algorithm = st.selectbox("Algorithm", options=list(MODELS.keys()))
        feature_engineering = st.selectbox(
            "Feature engineering", options=list(FEATURE_ENGINEERING.keys())
        )
        tune = st.checkbox("Tune hyperparameters", value=False)
        cross_validation = st.checkbox("Run 5-fold cross-validation", value=False)
        if tune:
            st.caption("Hyperparameter tuning can take several minutes, especially for Random Forest.")
    else:
        st.header("Prediction")
        st.caption("Select a saved model and upload unlabeled flow features.")

if workflow == "Predict new flows":
    render_prediction_workflow()
    st.stop()

st.caption("Training uses only the prepared sample derived from data/ml-data.zip.")
train_path = DATA_DIR / "train_mosaic.csv.zip"
test_path = DATA_DIR / "test_mosaic.csv.zip"
manifest_path = DATA_DIR / "ml-data_manifest.json"
if not all(path.is_file() for path in (train_path, test_path, manifest_path)):
    st.error("Prepared ml-data files are missing. Run `python prepare_ml_data.py` first.")
    st.stop()

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_archive") != "ml-data.zip":
        raise ValueError("The prepared files were not generated from data/ml-data.zip.")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
except Exception as error:
    st.error(f"Could not load prepared ml-data: {error}")
    st.stop()

st.success(
    f"Using ml-data only: {len(train_data):,} training and {len(test_data):,} test rows "
    f"with {len(train_data.columns) - 1} features."
)

if train_data is not None:
    if train_data.empty or len(train_data.columns) < 2:
        st.error("The training dataset needs at least one feature and a target column.")
        st.stop()

    target_column = "label"
    if target_column not in train_data.columns or target_column not in test_data.columns:
        st.error("The prepared ml-data files must contain the binary 'label' column.")
        st.stop()
    st.caption("Target label: `BENIGN` or `MALICIOUS` (derived from the ml-data archive).")
    render_data_visualizations(train_data, target_column)

    if st.button("Train and evaluate", type="primary", use_container_width=True):
        try:
            X, y = prepare_features(train_data, target_column)
            if y.nunique() < 2:
                raise ValueError("The target column must contain at least two classes.")

            X_train, y_train = X, y
            X_test, y_test = prepare_features(
                test_data, target_column, expected_columns=list(X.columns)
            )

            if y_train.nunique() < 2:
                raise ValueError("The training split must contain at least two target classes.")

            estimator, parameter_grid = MODELS[algorithm]
            pipeline = create_pipeline(
                clone(estimator), get_feature_engineering(feature_engineering)
            )
            best_params = None
            with st.spinner("Training and evaluating model..."):
                training_start = time.perf_counter()
                if tune:
                    pipeline, best_params = tune_hyperparameters(
                        pipeline, parameter_grid, X_train, y_train
                    )
                else:
                    pipeline.fit(X_train, y_train)
                training_time = time.perf_counter() - training_start

                cv_scores = None
                if cross_validation:
                    minimum_class_count = int(y_train.value_counts().min())
                    if minimum_class_count < 5:
                        st.warning(
                            "Skipping 5-fold cross-validation: each training class needs at least "
                            "5 examples."
                        )
                    else:
                        cv_scores = cross_val_score(
                            pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
                        )

                predictions = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions, output_dict=True)
                report_text = classification_report(y_test, predictions, digits=4)
                class_labels = list(pd.Series(y_test).unique())
                confusion = confusion_matrix(y_test, predictions, labels=class_labels)
                feature_importance = get_model_feature_importance(
                    pipeline, list(X_train.columns)
                )

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"ml_data_{algorithm.lower().replace(' ', '_')}_{timestamp}"
            model_filename = f"{base_name}_model.joblib"
            report_filename = f"{base_name}_report.txt"
            model_path = RESULTS_DIR / model_filename
            joblib.dump(pipeline, model_path)
            model_bytes = model_path.read_bytes()

            report_body = (
                f"Algorithm: {algorithm}\n"
                f"Feature engineering: {feature_engineering}\n"
                f"Hyperparameter tuning: {'Yes' if tune else 'No'}\n"
                f"Training rows: {len(X_train)}\n"
                f"Test rows: {len(X_test)}\n"
                f"Test accuracy: {accuracy:.4f}\n"
            )
            if cv_scores is not None:
                report_body += (
                    f"Cross-validation accuracy: {cv_scores.mean():.4f} "
                    f"(+/- {cv_scores.std():.4f})\n"
                )
            report_body += f"\nClassification report:\n{report_text}"
            report_path = RESULTS_DIR / report_filename
            report_path.write_text(report_body, encoding="utf-8")

            predictions_frame = X_test.reset_index(drop=True).copy()
            predictions_frame["actual"] = y_test.reset_index(drop=True)
            predictions_frame["predicted"] = predictions
            predictions_buffer = StringIO()
            predictions_frame.to_csv(predictions_buffer, index=False)

            st.session_state["last_run"] = {
                "accuracy": accuracy,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "training_time": training_time,
                "cv_scores": cv_scores,
                "best_params": best_params,
                "report_table": pd.DataFrame(report).transpose(),
                "confusion_table": pd.DataFrame(
                    confusion,
                    index=pd.Index([f"Actual {label}" for label in class_labels]),
                    columns=pd.Index([f"Predicted {label}" for label in class_labels]),
                ),
                "feature_importance": feature_importance,
                "model_bytes": model_bytes,
                "model_filename": model_filename,
                "report_text": report_body,
                "report_filename": report_filename,
                "predictions_csv": predictions_buffer.getvalue(),
            }
            st.success(f"Model and report saved in {RESULTS_DIR}.")
        except Exception as error:
            st.error(f"Could not complete model training: {error}")

if "last_run" in st.session_state:
    render_results(st.session_state["last_run"])

with st.expander("Dataset preview"):
    if train_data is not None:
        st.dataframe(train_data.head(10), use_container_width=True)
    else:
        st.caption("Choose a dataset source to preview data.")
