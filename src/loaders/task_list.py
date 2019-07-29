# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This file contains a list of all the tasks, their id and task name, description
and the tags associated with them.
"""

def task_exists(task):
    for entry in task_list:
        if entry['task']==task:
            return True
        else:
            return False

task_list = [
    {
        "id": "SemEval2017-3",
        "display_name": "SemEval 2017 - Task 3",
        "task": "Semeval",
        "tags": [ "All",  "QA" ],
        "description": "Subtask A: Question-Comment Similarity, Subtask B: Question-Question Similarity, Subtask C: Question-External Comment Similarity. From http://aclweb.org/anthology/S17-2003.",
        "notes": "Sentence similarity task"
    }
]
