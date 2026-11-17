"""Tests on the model: its answers must make sense to a person.

They test the model that train.py saved, model.joblib: the same model that the
quality gate checks and that a release publishes. Run them after train.py.
"""
import joblib
import pandas as pd

model = joblib.load("model.joblib")
FEATURES = ["size_m2", "rooms", "age_years"]


def price(size, rooms, age):
    house = pd.DataFrame([[size, rooms, age]], columns=FEATURES)
    return model.predict(house)[0]


def test_typical_house_has_a_plausible_price():
    assert 150 < price(100, 3, 15) < 400


def test_bigger_house_costs_more():
    assert price(150, 3, 15) > price(100, 3, 15)


def test_older_house_costs_less():
    assert price(100, 3, 40) < price(100, 3, 5)
