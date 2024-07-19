import data_management as dm
import config

def make_prediction(*, path_to_images) -> float:
    pipe = dm.load_image_paths()
    predictions = pipe.pipe.predict(path_to_images)

    return predictions

if __name__=='__main__':
    import joblib

    images_df = dm.load_image_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)

    pipe = joblib.load(config.PIPELINE_PATH)

    predictions = pipe.predict(X_test)
    print(predictions)