import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

"""
Helper functions for scoring accoring to the competition evaluation metric.
https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/overview/evaluation
"""


def check_submission(sub):
    """
    Checks that the submission meets all the requirements.
    1. No more than 22 Boxes per frame.
    2. Only one label prediction per video/frame
    3. No duplicate boxes per frame.
    4. Boxes must be within video area:
        - `top` and `left` must each be >= 0
        - The sum of `left` and `width` must be <= 1280
        - The sum of `top` and `height` must be must be <= 720
    Args:
        sub : submission dataframe.
    Returns:
        True -> Passed the tests
        False -> Failed the test
    """
    # Maximum of 22 boxes per frame.
    max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
    if max_box_per_frame > 22:
        print("Has more than 22 boxes in a single frame")
        return False
    # Only one label allowed per frame.
    has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
    if has_duplicate_labels:
        print("Has duplicate labels")
        return False
    # Check for unique boxes
    has_duplicate_boxes = sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
    if has_duplicate_boxes:
        print("Has duplicate boxes")
        return False
    if sub['left'].min() < 0:
        print('left column has values less than 0')
        return False
    if sub['top'].min() < 0:
        print('top column has values less than 0')
        return False
    if (sub['left'] + sub['width']).max() > 1280:
        print('left+width columns has values greater than 1280')
        return False
    if (sub['top'] + sub['height']).max() > 720:
        print('top+height columns has values greater than 720')
        return False
    return True

def force_sub_requirements(sub, verbose=True):
    """
    Enforces the submission submission
    *Warning* Using this code may remove prediction rows in a sub-optimal manner.
    """
    len_before = len(sub)
    sub = sub.drop_duplicates(['video_frame', 'left','width','top','height'])
    sub = sub.drop_duplicates(['video_frame', 'label'])
    sub = sub.groupby("video_frame").head(22)
    sub = sub.loc[sub['left'] >= 0]
    sub = sub.loc[sub['top'] >= 0]
    sub = sub.loc[(sub['top'] + sub['height']) <= 720]
    sub = sub.loc[(sub['left'] + sub['width']) <= 1280]
    sub = sub.reset_index(drop=True)
    if verbose:
        len_after = len(sub)
        n_removed = len_before - len_after
        print(f'Forcing submission requirements removed {n_removed} rows ({len_before} -> {len_after})')
    return sub

class NFLAssignmentScorer:
    def __init__(
        self,
        labels_df: pd.DataFrame = None,
        labels_csv="train_labels.csv",
        check_constraints=True,
        weight_col="isDefinitiveImpact",
        impact_weight=1000,
        iou_threshold=0.35,
        remove_sideline=True,
    ):
        """
        Helper class for grading submissions in the
        2021 Kaggle Competition for helmet assignment.
        Version 1.1
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
        Use:
        ```
        scorer = NFLAssignmentScorer(labels)
        scorer.score(submission_df)
        or
        scorer = NFLAssignmentScorer(labels_csv='labels.csv')
        scorer.score(submission_df)
        ```
        Args:
            labels_df (pd.DataFrame, optional):
                Dataframe containing theground truth label boxes.
            labels_csv (str, optional): CSV of the ground truth label.
            check_constraints (bool, optional): Tell the scorer if it
                should check the submission file to meet the competition
                constraints. Defaults to True.
            weight_col (str, optional):
                Column in the labels DataFrame used to applying the scoring
                weight.
            impact_weight (int, optional):
                The weight applied to impacts in the scoring metrics.
                Defaults to 1000.
            iou_threshold (float, optional):
                The minimum IoU allowed to correctly pair a ground truth box
                with a label. Defaults to 0.35.
            remove_sideline (bool, optional):
                Remove slideline players from the labels DataFrame
                before scoring.
        """
        if labels_df is None:
            # Read label from CSV
            if labels_csv is None:
                raise Exception("labels_df or labels_csv must be provided")
            else:
                self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = labels_df.copy()
        if remove_sideline:
            self.labels_df = self.labels_df.query("isSidelinePlayer == False").reset_index(drop=True).copy()
        self.impact_weight = impact_weight
        self.check_constraints = check_constraints
        self.weight_col = weight_col
        self.iou_threshold = iou_threshold

    def add_xy(self, df):
        """
        Adds `x1`, `x2`, `y1`, and `y2` columns necessary for computing IoU.
        Note - for pixel math, 0,0 is the top-left corner so box orientation
        defined as right and down (height)
        """

        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, sub, labels, weight_col="isDefinitiveImpact"):
        """
        Perform an outer join between submission and label.
        Creates a `sub_label` dataframe which stores the matched label for each submission box.
        Ground truth values are given the `_gt` suffix, submission values are given `_sub` suffix.
        """
        sub = sub.copy()
        labels = labels.copy()

        sub = self.add_xy(sub)
        labels = self.add_xy(labels)

        base_columns = ["label", "video_frame", "x1", "x2", "y1", "y2", "left", "width", "top", "height"]

        sub_labels = sub[base_columns].merge(
            labels[base_columns + [weight_col]], on=["video_frame"], how="right", suffixes=("_sub", "_gt")
        )
        return sub_labels

    def get_iou_df(self, df):
        """
        This function computes the IOU of submission (sub)
        bounding boxes against the ground truth boxes (gt).
        """
        df = df.copy()

        # 1. get the coordinate of inters
        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)

        # 2. calculate the area of inters
        df["inters"] = df["iw"] * df["ih"]

        # 3. calculate the area of union
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) * (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        # print(uni)
        # 4. calculate the overlaps between pred_box and gt_box
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1)

    def filter_to_top_label_match(self, sub_labels):
        """
        Ensures ground truth boxes are only linked to the box
        in the submission file with the highest IoU.
        """
        return sub_labels.sort_values("iou", ascending=False).groupby(["video_frame", "label_gt"]).first().reset_index()

    def add_isCorrect_col(self, sub_labels):
        """
        Adds True/False column if the ground truth label
        and submission label are identical
        """
        sub_labels["isCorrect"] = (sub_labels["label_gt"] == sub_labels["label_sub"]) & (
            sub_labels["iou"] >= self.iou_threshold
        )
        return sub_labels

    def calculate_metric_weighted(self, sub_labels, weight_col="isDefinitiveImpact", weight=1000):
        """
        Calculates weighted accuracy score metric.
        """
        sub_labels["weight"] = sub_labels.apply(lambda x: weight if x[weight_col] else 1, axis=1)
        y_pred = sub_labels["isCorrect"].values
        y_true = np.ones_like(y_pred)
        weight = sub_labels["weight"]
        return accuracy_score(y_true, y_pred, sample_weight=weight)

    def score(self, sub, labels_df=None, drop_extra_cols=True):
        """
        Scores the submission file against the labels.
        Returns the evaluation metric score for the helmet
        assignment kaggle competition.
        If `check_constraints` is set to True, will return -999 if the
            submission fails one of the submission constraints.
        """
        if labels_df is None:
            labels_df = self.labels_df.copy()

        if self.check_constraints:
            if not check_submission(sub):
                return -999
        sub_labels = self.merge_sub_labels(sub, labels_df, self.weight_col)
        sub_labels = self.get_iou_df(sub_labels).copy()
        sub_labels = self.filter_to_top_label_match(sub_labels).copy()
        sub_labels = self.add_isCorrect_col(sub_labels)
        score = self.calculate_metric_weighted(sub_labels, self.weight_col, self.impact_weight)
        # Keep `sub_labels for review`
        if drop_extra_cols:
            drop_cols = ["x1_sub", "x2_sub", "y1_sub", "y2_sub", "x1_gt", "x2_gt", "y1_gt", "y2_gt"]
            sub_labels = sub_labels.drop(drop_cols, axis=1)
        self.sub_labels = sub_labels
        return score

    
if __name__ == "__main__":
    sub = pd.DataFrame(columns=['isSidelinePlayer'])
    scorer = NFLAssignmentScorer(sub)
