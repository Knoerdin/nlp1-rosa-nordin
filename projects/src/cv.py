"""Cross-validation helpers on top of the course code (part1.experiment)."""
from collections import OrderedDict

from tqdm.auto import tqdm

from part1 import experiment


def course_cv(data_splits, outcome_space, encoder, model_option, save_to, **kwargs):
    """
    part1.experiment.cross_validation_experiment (model_option "logistic" or "mlp"), but
    `encoder` may also be a {fold: encoder} dict, e.g. from encoders.fit_per_fold, so that
    fitted representations (TF-IDF) only ever see the training part of their own fold.
    Returns the same dict, so results_io.save_cv works unchanged.
    """
    if not isinstance(encoder, dict):
        return experiment.cross_validation_experiment(data_splits, outcome_space, encoder, model_option,
                                                      save_to=save_to, **kwargs)
    out = {key: OrderedDict() for key in ("models", "predictions", "accuracies")}
    for fold in tqdm(data_splits):
        # one fold at a time, each with its own encoder (same model paths: {save_to}/fold-{k}/model)
        one = experiment.cross_validation_experiment({fold: data_splits[fold]}, outcome_space, encoder[fold],
                                                     model_option, save_to=save_to, **kwargs)
        for key in out:
            out[key].update(one[key])
    return out
