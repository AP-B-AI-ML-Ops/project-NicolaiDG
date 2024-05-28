import pandas as pd
import pytest

from load.model_prep import split_train_test_val


@pytest.fixture
def dataset():
    data = {"Feature1": range(10), "Feature2": range(10, 20), "Quality": [0, 1] * 5}
    return pd.DataFrame(data)


def test_split_train_test_val(dataset):
    # Roep de functie aan die we willen testen
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val(dataset)

    # Controleer de grootte van de gesplitste datasets
    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(X_val) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2
    assert len(y_val) == 2

    # Controleer of de splitsing consistent is
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_val) == len(y_val)

    # Controleer of de gegevens correct zijn gesplitst
    train_indices = X_train.index
    test_indices = X_test.index
    val_indices = X_val.index

    all_indices = set(train_indices).union(set(test_indices)).union(set(val_indices))
    assert len(all_indices) == 10  # Zorg ervoor dat alle indices aanwezig zijn
    assert (
        set(range(10)) == all_indices
    )  # Zorg ervoor dat alle oorspronkelijke indices zijn opgenomen
