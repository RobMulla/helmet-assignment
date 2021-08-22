# NFL Helmet Assignment Helpers
A package of code to assist in the 2021 Kaggle NFL Helmet Assignment Task

## Install
```bash
$ git clone https://github.com/RobMulla/helmet-assignment.git
$ cd helmet-assignment
$ pip install .
```

## Scoring
This code can be used to score your predictions in a similar to the offical competition metric.

```python
from helmet_assingment.score import NFLAssignmentScorer
scorer = NFLAssignmentScorer(labels)
scorer.score(submission_df)

or

scorer = NFLAssignmentScorer(labels_csv='labels.csv')
scorer.score(submission_df)

```

The `check_submission` can be used as a final check to ensure your submission meets all the requirements of the submission:

```python
check_submission(submission_df)
>> True # If passed otherwise returns False
```

## Videos
Code here can be used to create videos that display your predictions against ground truth boxes.

The `video_with_predictions` function allows you to combine the results from the `NFLAssignmentScorer` and overlay the results in video format.

## Features

Theo code contains helper functions which add features to the data.

`add_track_features` adds additional features to the tracking data which can help when attempting to merge this data onto the video frames.
