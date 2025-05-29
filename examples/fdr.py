import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import functools
    import ast

    import jax
    import jax.numpy as jnp

    import pandas as pd

    import altair as alt
    alt.data_transformers.enable("vegafusion")

    from winnow.fdr.database_grounded import DatabaseGroundedFDRControl
    from winnow.fdr.bayes import EmpiricalBayesFDRControl, Distribution

    from winnow.datasets.calibration_dataset import RESIDUE_MASSES
    return alt, ast, jax, jnp, mo, pd


@app.cell
def _(mo):
    dataset = mo.ui.dropdown(
        label="Dataset",
        options=[
            "Hela QC", "S. Brodae", "GluC", "Herceptin", "Snake Venoms", "Immunopeptidomics"
        ],
        value="Hela QC"
    )
    n_steps = mo.ui.number(label="No. updates", start=5000, stop=200_000, step=5000)
    confidence_type = mo.ui.dropdown(label="Confidence type", options=["Raw", "Calibrated"], value="Calibrated")
    SPECIES_DICT = {
        "Hela QC": "helaqc", "S. Brodae": "sbrodae", "GluC": "gluc", "Herceptin": "herceptin",
        "Snake Venoms": "snakevenoms", "Immunopeptidomics": "immuno"
    }
    mo.hstack([dataset, confidence_type, n_steps])
    return SPECIES_DICT, confidence_type, dataset


@app.cell
def _(SPECIES_DICT, ast, dataset, pd):
    test_dataset_metadata = pd.read_csv(
        f"/Users/amandlamabona/Projects/winnow/calibrated_datasets/labelled/{SPECIES_DICT[dataset.value]}_test_labelled.csv"
    )

    def try_convert(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value  # Return original value if conversion fails


    # Apply conversion to all object (string) columns
    for col in test_dataset_metadata.select_dtypes(include=["object"]).columns:
        test_dataset_metadata[col] = test_dataset_metadata[col].apply(try_convert)
    return (test_dataset_metadata,)


@app.cell
def _(alt, confidence_type, mo, test_dataset_metadata):
    confidence_column = "confidence" if confidence_type.value == "Raw" else "calibrated_confidence"
    plot = alt.Chart(test_dataset_metadata).mark_bar(opacity=0.7).encode(
        x=alt.X(confidence_column).bin(step=0.01),
        y=alt.Y("count()").stack(None),
        color="correct"
    )
    mo.ui.altair_chart(plot)
    return (confidence_column,)


@app.cell
def _(confidence_column, test_dataset_metadata):
    confidence = test_dataset_metadata[confidence_column].sort_values(ascending=False)
    return (confidence,)


@app.cell
def _(alt, jnp, pd, test_dataset_metadata):
    def get_plot_dataframe(input_df: pd.DataFrame, confidence_column: str, fdr_function) -> pd.DataFrame:
        sorted_df = input_df.sort_values(ascending=False, by=[confidence_column])
        cum_correct = jnp.cumsum(jnp.array(sorted_df['correct']))
        cum_counts = jnp.arange(1, len(test_dataset_metadata) + 1)
        true_fdr = (cum_counts - cum_correct)/cum_counts
        estimated_fdr = fdr_function(sorted_df[confidence_column])
        multi_plot_df = pd.DataFrame({
            'confidence': pd.concat([sorted_df[confidence_column], sorted_df[confidence_column]]) ,
            'fdr': true_fdr.tolist() + estimated_fdr.tolist(),
            'source': true_fdr.shape[0]*['true'] + estimated_fdr.shape[0]*['estimate']
        })
        return multi_plot_df

    def get_confidence_threshold(dataframe: pd.DataFrame) -> float:
        sorted_df = dataframe.sort_values(by=['confidence'])
        idxs = jnp.where(jnp.diff(jnp.sign(jnp.array(sorted_df['fdr']) - 0.05)) != 0)
        return sorted_df['confidence'].values[idxs[0][0] + 1].item()

    def plot_fdr_accuracy(title: str, input_df: pd.DataFrame, confidence_column: str, fdr_function) -> alt.Chart:
        multi_plot_df = get_plot_dataframe(input_df=input_df, confidence_column=confidence_column, fdr_function=fdr_function)
        cutoffs = multi_plot_df.groupby('source').apply(get_confidence_threshold).to_frame(name='value').reset_index()
        cutoff_plots = alt.Chart(cutoffs).mark_rule(strokeDash=[4, 4]).encode(x='value:Q', color='source')
        line_plot = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(y=alt.datum(0.05)).properties(title=title)
        fdr_plot = alt.Chart(multi_plot_df).mark_line().encode(x='confidence', y='fdr', color='source')

        return fdr_plot + line_plot + cutoff_plots
    return (plot_fdr_accuracy,)


@app.cell
def _(
    confidence_column,
    confidence_type,
    dataset,
    mo,
    nonparametric_calibrated_estimator,
    plot_fdr_accuracy,
    test_dataset_metadata,
):
    mo.ui.altair_chart(plot_fdr_accuracy(
        title=f'Nonparametric Calibrated FDR Estimator: {dataset.value}, {confidence_type.value}',
        input_df=test_dataset_metadata,
        confidence_column=confidence_column,
        fdr_function=nonparametric_calibrated_estimator
    ))
    return


@app.cell
def _(alt, dataset, jnp, mo, pd, test_dataset_metadata):
    pr_df = test_dataset_metadata[['confidence', 'calibrated_confidence', 'correct']]
    def compute_pr_curve(column: str):
        sorted_df = pr_df.sort_values(ascending=False, by=[column])
        cum_correct = jnp.cumsum(jnp.array(sorted_df['correct']))
        cum_counts = jnp.arange(1, len(sorted_df) + 1)
        precision = cum_correct / cum_counts
        recall = cum_correct / len(sorted_df)
        return pd.DataFrame(
            {'precision': precision.tolist(),
            'recall': recall.tolist(),
            'source': len(sorted_df)*[column]}
        )

    pr_curve_df = pd.concat([
            compute_pr_curve('confidence'), compute_pr_curve('calibrated_confidence')
    ])
    pr_plot = alt.Chart(pr_curve_df).mark_line().encode(
        x='recall', y=alt.Y('precision').scale(domain=(pr_curve_df['precision'].min(), 1.0)),
        color='source'
    )
    line_plot = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(y=alt.datum(0.95))
    mo.ui.altair_chart((pr_plot + line_plot).properties(title=f"Precision-Recall Curve: {dataset.value}"))
    return (pr_curve_df,)


@app.cell
def _(pr_curve_df):
    first, second = pr_curve_df.groupby('source')
    first
    return


@app.cell
def _(jnp):
    def nonparametric_calibrated_estimator(probabilities):
        error_probabilities = jnp.array(1 - probabilities)
        counts = jnp.arange(1, len(error_probabilities) + 1)
        cum_error_probabilities = jnp.cumsum(error_probabilities)
        false_discovery_rate = cum_error_probabilities / counts
        return false_discovery_rate
    return (nonparametric_calibrated_estimator,)


@app.cell
def _(confidence, nonparametric_calibrated_estimator):
    fdr = nonparametric_calibrated_estimator(probabilities=confidence)
    return


@app.cell
def _():
    import bisect
    return


@app.cell
def _(test_dataset_metadata):
    1- len(test_dataset_metadata[test_dataset_metadata['correct']])/len(test_dataset_metadata)
    return


@app.cell
def _(jax, jnp):
    def update_posterior(posteriors: jax.Array, prior_train: float, prior_test: float) -> float:
        adapted_posteriors = (
            ((prior_test/prior_train) * posteriors) / 
            (((prior_test/prior_train) * posteriors)  + (((1 - prior_test)/(1 - prior_train)) * (1 - posteriors)))
        )
        return adapted_posteriors

    def update_priors(posteriors: jax.Array) -> float:
        return jnp.mean(posteriors).item()
    return


if __name__ == "__main__":
    app.run()
