from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    zip_file_path = r"data\raw\archive.zip"  # or "data/raw/archive.zip"

    selected_features = [
        "cycle", "setting_1", "setting_2", "setting_3",
        "s2", "s3", "s4", "s7", "s8", "s9"
    ]

    training_pipeline(
        file_path=zip_file_path,
        subset="FD001",
        part="train",
        missing_value_strategy="mean",
        fe_strategy="standard_scaling",
        features=selected_features,
    )
