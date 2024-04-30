from pathlib import Path

from prodigy.components.loaders import JSONL
from prodigy.core import Arg, recipe
from prodigy.util import set_hashes

options = [
    {
        "id": "Précis",
        "text": "Précis (entre 6 et 10)",
        "style": {"background": "#36FF33"},
    },
    {
        "id": "Pertinent",
        "text": "Pertinence thématique (entre 2 et 5)",
        "style": {"background": "#FFF12E"},
    },
    {"id": "Off-track", "text": "Hors sujets (1)", "style": {"background": "#FF0800"}},
]

root = Path(__file__).parent


# Helper functions for adding user provided labels to annotation tasks.
def add_label_options_to_stream(stream):
    for task in stream:
        task["options"] = options
        task["original"] = task["text"]
        task["simplified"] = task["simplified_text"]
        yield task


@recipe(
    "legal_meaning",
    dataset=Arg(help="Dataset to save answers to."),
    source=("The source data as a JSONL file", "positional", None, str),
)
def legal_meaning(dataset: str, source: str):
    def validate_answer(eg):
        high_level_annotation = eg.get("accept", [])
        assert (
            1 <= len(high_level_annotation) <= 3
        ), "Sélectionner l'une des trois régions où le texte généré se situe."
        region_level_annotation = eg.get("evaluation", None)
        assert region_level_annotation is not None, (
            "Sélectionner le niveau sur l'échelle de 1 à 10 où "
            "le texte généré se situe."
        )
        if "." in region_level_annotation:
            raise ValueError(
                "Sélectionner un nombre entier (p. ex. 1, 2, 3) pour le positionnement sur l'échelle."
            )
        try:
            region_level_annotation = int(region_level_annotation)
        except ValueError:
            raise ValueError("Inscrire un nombre pour le positionnement sur l'échelle.")
        assert 10 >= region_level_annotation >= 1, (
            "Inscrire un nombre entre 1 et 10 pour le positionnement " "sur l'échelle."
        )

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)
    stream = list(
        (set_hashes(eg, input_keys=("text", "simplified_text")) for eg in stream)
    )

    # Add labels to each task in stream
    stream = add_label_options_to_stream(stream)
    blocks = [
        {"view_id": "html", "html_template": (root / "recipe.html").read_text()},
        {
            "view_id": "text_input",
            "field_rows": 3,
            "field_placeholder": "Tapez ici...",
            "field_label": "Identifier le niveau de difficulté selon les choix suivants:",
            "field_id": "difficulty_evaluation",
            "field_suggestions": [
                "Livre pour enfants",
                "Recette de cuisine",
                "Article de nouvelle",
                "Article Wikipédia",
                "Nouvelle ou roman",
                "Dictée",
                "Contrat d'assurance",
                "Document légal",
            ],
        },
        {
            "view_id": "html",
            "html_template": "Étape 2 de l'évaluation: Selon vous, quel est le niveau de précision légale du "
            "texte généré par le modèle selon les trois choix suivants?",
        },
        {
            "view_id": "choice",
            "text": None,
            "field_id": "meaning_evaluation",
        },
        {
            "view_id": "html",
            "html_template": "Étape 3 de l'évaluation: Selon votre réponse précédente sur l'échelle de "
            "précision, indiquer la valeur exacte du niveau de précision légale du texte "
            "généré par le modèle sur l'échelle appropriée?",
        },
        {
            "view_id": "text_input",
            "field_rows": 1,
            "field_label": "Identifier le niveau de précision légale selon les choix suivants:",
            "field_placeholder": "Tapez ici...",
            "field_id": "evaluation",
            "field_suggestions": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        },
        {"view_id": "html", "html_template": "Commentaires (si applicable):"},
        {
            "view_id": "text_input",
            "field_rows": 3,
            "field_placeholder": "Tapez ici...",
            "field_label": "Commentaires sur le texte ou l'annotation (si applicable).",
            "field_id": "comments",
        },
    ]

    return {
        "view_id": "blocks",
        "dataset": dataset,  # save annotations in this dataset
        "stream": stream,
        "config": {  # Additional config settings, mostly for app UI
            "exclude_by": "input",  # Hash value used to filter out already seen examples
            "blocks": blocks,
        },
        "validate_answer": validate_answer,
    }
