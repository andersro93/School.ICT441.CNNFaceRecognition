#!/bin/python3
from data import Data
from gender_classifier import GenderClassifier
from individual_classifier import IndividualClassifier

if __name__ == '__main__':

    # Ensure that the data is in place
    Data().ensure_tmp_is_created_and_structured()

    # Run gender classifier
    print(f"Running gender classification")
    gender_classifier = GenderClassifier()
    gender_classifier.create_model()
    gender_classifier.create_train_verification_data()
    gender_classifier.fit_model()

    # Run individual classifier
    print(f"Running individual classification")
    individual_classifier = IndividualClassifier()
    individual_classifier.create_model()
    individual_classifier.create_train_verification_data()
    individual_classifier.fit_model()
