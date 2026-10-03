"""Reuse the complete-cohort paper reducer for the frozen kernel-pinned follow-up.

Requires a terminal336 parent receipt and independent review. No partial output,
new criteria, GPU execution, scientific gate or manuscript mutation.
"""
from pathlib import Path
import hashlib
import build_native_joint_paper_result_v1 as base

SOURCE_SHA = '92d14b5d56867eb68d685830acdf05b09021992194a7208ebdcd3809d68898be'
source = Path(base.__file__)
base.require(hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA,
             'reviewed complete-cohort paper reducer changed')
base.CORPUS = base.C / 'fullmodel/native-joint-kernel-pin-retry1-v1/CORPUS.json'
base.CORPUS_SHA = '8b1ab3e055292801e1e40b3f6372b49bd48377f30d60f6eb5df01e74115159ae'
original_render = base.render


def render(result):
    text = original_render(result)
    return text.replace(
        'A separate reference experiment imports both target and MTP starting state and',
        'A follow-up reference experiment pins the native kernel-selection cache,\n'
        'imports the same target and MTP starting state, and')


if __name__ == '__main__':
    base.render = render
    base.main()
