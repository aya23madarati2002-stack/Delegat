from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAINING_EXAMPLES = [
    # Utility Bill / Rechnung
    (
        "Rechnung Rechnungsdatum Gesamtbetrag Stromverbrauch Energie Anbieter Kunde Verbrauchsrechnung",
        "utility_bill",
    ),
    (
        "Invoice date total amount electricity gas water utility provider customer billing address",
        "utility_bill",
    ),
    (
        "Vattenfall Energie Rechnung Strom Gesamtbetrag Rechnungsnummer Kunde Verbrauch",
        "utility_bill",
    ),
    (
        "E.ON Stromrechnung Verbrauchsabrechnung Zählernummer Rechnungsbetrag Rechnungsdatum",
        "utility_bill",
    ),
    (
        "Berliner Wasserbetriebe Rechnung Wasserverbrauch Kundennummer Verbrauchsstelle Rechnungsbetrag",
        "utility_bill",
    ),

    # Bank Statement / Kontoauszug
    (
        "Kontoauszug IBAN Kontostand Buchung Soll Haben Überweisung Bank statement balance transaction",
        "bank_statement",
    ),
    (
        "Bank statement account number transaction balance debit credit payment reference",
        "bank_statement",
    ),
    (
        "Sparkasse Kontoauszug Kontonummer IBAN BIC Buchungstag Valuta Saldo",
        "bank_statement",
    ),
    (
        "Commerzbank account statement opening balance closing balance payment transfer",
        "bank_statement",
    ),

    # Passport / Ausweis
    (
        "Passport surname given names nationality date of birth document number expiry date",
        "passport",
    ),
    (
        "Personalausweis Ausweisnummer Geburtsdatum Staatsangehörigkeit gültig bis Name Vorname",
        "passport",
    ),
    (
        "Identity card passport document number nationality birth place expiry authority",
        "passport",
    ),

    # Handelsregister
    (
        "Handelsregister Amtsgericht HRB Geschäftsführer Gesellschaft mit beschränkter Haftung GmbH",
        "commercial_register",
    ),
    (
        "Commercial register company registration number managing director legal form registered office",
        "commercial_register",
    ),
    (
        "Registerauszug HRB Stammkapital Sitz der Gesellschaft Vertretungsberechtigter Geschäftsführer",
        "commercial_register",
    ),

    # Unknown / Sonstiges
    (
        "random text document without invoice data no company validation no relevant fields",
        "unknown",
    ),
    (
        "presentation slides project description agenda meeting notes university prototype",
        "unknown",
    ),
    (
        "email conversation appointment calendar university homework unrelated content",
        "unknown",
    ),
]


@lru_cache(maxsize=1)
def get_document_classifier() -> Pipeline:
    texts = [example[0] for example in TRAINING_EXAMPLES]
    labels = [example[1] for example in TRAINING_EXAMPLES]

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(texts, labels)

    return model


def classify_document(text: str) -> dict:
    model = get_document_classifier()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    class_probabilities = {
        label: round(float(probability), 3)
        for label, probability in zip(classes, probabilities)
    }

    confidence = round(max(probabilities) * 100)

    return {
        "predicted_type": prediction,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }


def map_ml_type_to_extractor_type(predicted_type: str) -> str:
    if predicted_type == "utility_bill":
        return "utility_invoice"

    return predicted_type